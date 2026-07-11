"""Construct simulation engines and components from YAML configuration."""

import random
from pathlib import Path
from typing import Any

import yaml

from smartevsim.engine import Engine
from smartevsim.scenario_generator import ScenarioGenerator
from smartevsim.units.central_controller import (
    CentralController,
    ConstantCentralController,
    OptimalCentralController,
    PIConstantCentralController,
)
from smartevsim.units.charger import ExternalCharger, IntegratedCharger
from smartevsim.units.ev import EV
from smartevsim.units.micro_controller import (
    CentralizedMicroController,
    MicroController,
    RuleMicroController,
)

from smartevsim.units.pcc import PCC
from smartevsim.units.server import Server
from smartevsim.utils.data_classes import PriorityConfig, SimConfig

CENTRAL_CONTROLLER_MAP: dict = {
    "ConstantCentralController": ConstantCentralController,
    "PIConstantCentralController": PIConstantCentralController,
    "OptimalCentralController": OptimalCentralController,
}
MICRO_CONTROLLER_MAP: dict = {
    "RuleMicroController": RuleMicroController,
    "CentralizedMicroController": CentralizedMicroController,
}
CHARGER_MAP: dict = {
    "IntegratedCharger": IntegratedCharger,
    "ExternalCharger": ExternalCharger,
}

def register_micro_controller(name: str, controller_class: type[MicroController]) -> None:
    if name in MICRO_CONTROLLER_MAP:
        raise ValueError(f"Micro-controller type {name!r} is already registered")
    MICRO_CONTROLLER_MAP[name] = controller_class


def register_central_controller(name: str, controller_class: type[CentralController]) -> None:
    if name in CENTRAL_CONTROLLER_MAP:
        raise ValueError(f"Central-controller type {name!r} is already registered")
    CENTRAL_CONTROLLER_MAP[name] = controller_class


def build_charger_data_from_config(
    charger_cfg: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    """Resolve charger defaults and merge explicit configuration values.

    Args:
        charger_cfg: Charger parameters, optionally naming a defaults file.
        base_dir: Root configuration directory.

    Returns:
        Fully resolved charger constructor arguments.
    """
    charger_data = dict(charger_cfg)
    # If a charger config file is specified, load it and use it as defaults for the charger
    # parameters, overridden by any explicit charger fields.
    charger_config_name = charger_data.pop("charger_config", None)
    if charger_config_name:
        charger_config_path = Path(charger_config_name)
        if not charger_config_path.is_absolute():
            charger_config_path = base_dir / "charger_configs" / charger_config_path
        with open(charger_config_path, encoding="utf-8") as charger_file:
            charger_defaults = yaml.safe_load(charger_file) or {}
        if not isinstance(charger_defaults, dict):
            msg = (
                f"Charger config file {charger_config_path} must contain a mapping of "
                "charger parameters."
            )
            raise TypeError(msg)
        # Defaults from file, overridden by explicit charger fields.
        charger_data = {**charger_defaults, **charger_data}
    return charger_data


def build_ev_data_from_config(ev_cfg: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolve EV defaults and merge explicit per-EV configuration values.

    Args:
        ev_cfg: EV parameters, optionally naming an ``ev_config`` defaults
            file. Explicit values override defaults.
        base_dir: Root configuration directory containing ``ev_configs``.

    Returns:
        Fully resolved EV constructor arguments.

    Raises:
        TypeError: If the defaults file does not contain a mapping.
    """
    ev_data = dict(ev_cfg)
    # if an EV config file is specified, load it and use it as defaults for the EV parameters,
    # overridden by any explicit per-EV fields.
    ev_config_name = ev_data.pop("ev_config", None)
    if ev_config_name:
        ev_config_path = Path(ev_config_name)
        if not ev_config_path.is_absolute():
            ev_config_path = base_dir / "ev_configs" / ev_config_path
        with open(ev_config_path, encoding="utf-8") as ev_file:
            ev_defaults = yaml.safe_load(ev_file) or {}
        if not isinstance(ev_defaults, dict):
            msg = f"EV config file {ev_config_path} must contain a mapping of EV parameters."
            raise TypeError(msg)
        # Defaults from file, overridden by explicit per-EV fields.
        ev_data = {**ev_defaults, **ev_data}
    return ev_data

def build_engine_from_yaml(path: str) -> Engine:
    """Build a fully wired simulation engine from a YAML scenario file.

    Args:
        path: Path to the scenario YAML file.

    Returns:
        Configured engine with all requested simulation components.

    Raises:
        KeyError: If a required configuration section or registered component
            type is missing.
    """
    base_dir = Path(__file__).resolve().parent / "config"
    # base_dir = Path("src/smartevsim/config")
    with open(path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    sim_config = SimConfig(**cfg["sim_config"])
    priority_config = PriorityConfig(**cfg["priority_config"])
    seed_rng = random.Random(sim_config.seed)

    server = Server()

    cc_cfg = cfg["central_controller"]
    cc_cls = CENTRAL_CONTROLLER_MAP[cc_cfg["type"]]
    central_controller = cc_cls(**cc_cfg["params"], seed=seed_rng.randrange(2**32))

    mc_cfg = cfg["micro_controllers"]
    mc_cls = MICRO_CONTROLLER_MAP[mc_cfg["type"]]

    charger_cfg = cfg["chargers"]
    charger_cls = CHARGER_MAP[charger_cfg["type"]]
    charger_params = build_charger_data_from_config(charger_cfg["params"], base_dir)

    if "ev_generation" in cfg:
        scenario_generator = ScenarioGenerator(
            **cfg["ev_generation"],
            sim_config=sim_config,
            seed=seed_rng.randrange(2**32),
        )
        cfg["evs"] = scenario_generator.generate_ev_configs()

    evs: list[EV] = []
    mcs: list = []
    chargers: list = []
    for ev_cfg in cfg["evs"]:
        ev_params = build_ev_data_from_config(ev_cfg, base_dir)
        evs.append(EV(server, **ev_params))
        ev_name = ev_params["name"]
        mc_name = f"mc_{ev_name}"
        mcs.append(
            mc_cls(
                **mc_cfg["params"],
                name=mc_name,
                ev_name=ev_name,
                seed=seed_rng.randrange(2**32),
            )
        )
        charger_name = f"charger_{ev_name}"
        chargers.append(charger_cls(
            **charger_params, name=charger_name, ev_name=ev_name, mc_name=mc_name
        ))

    pcc = PCC(**cfg["pcc"], seed=seed_rng.randrange(2**32))

    return Engine(
        config=sim_config,
        priority_config=priority_config,
        server=server,
        pcc=pcc,
        central_controller=central_controller,
        micro_controllers=mcs,
        chargers=chargers,
        evs=evs,
    )
