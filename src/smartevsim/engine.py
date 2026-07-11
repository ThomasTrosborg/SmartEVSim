
"""Core simulation orchestration and component interface definitions."""

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from smartevsim.units.server import Server
from smartevsim.utils.data_classes import SimConfig, StepRecord, WorldState

pd.options.plotting.backend = "plotly"

logger = logging.getLogger(__name__)


class PCCProtocol(Protocol):
    """Interface required of a point-of-common-coupling component.

    Attributes:
        name: Unique component name.
        p_total_kw: Latest aggregate power measurement.
    """
    name: str
    p_total_kw: float

    def step(self, engine: "Engine") -> None:
        """Update the aggregate measurement using the engine state."""
        ...


class CentralControllerProtocol(Protocol):
    """Interface required of a central controller component.

    Attributes:
        name: Unique controller name.
        p_setpoint: Current aggregate power reference.
    """
    name: str
    p_setpoint: float

    def step(self, engine: "Engine", t: float) -> None:
        """Update central control outputs at simulated time *t*."""
        ...


class MicroControllerProtocol(Protocol):
    """Interface required of a per-EV micro-controller component.

    Attributes:
        ev_name: Associated EV name, when assigned.
        p_setpoint: Current per-EV power setpoint.
    """
    ev_name: str | None
    p_setpoint: float

    def step(self, engine: "Engine") -> None:
        """Update the per-EV controller from the engine state."""
        ...


class EVProtocol(Protocol):
    """Interface required of an electric-vehicle component.

    Attributes:
        name: Unique EV name.
        p_cons_kw: Current power consumption.
        next_p_cons_kw: Calculated next-step consumption.
    """
    name: str
    p_cons_kw: float
    next_p_cons_kw: float

    def step(self, engine: "Engine", dt: float) -> None:
        """Advance EV state by a step of *dt* seconds."""
        ...


class ChargerProtocol(Protocol):
    """Interface required of a charger component.

    Attributes:
        ev_name: Associated EV name.
        p_setpoint: Current EV-facing power setpoint.
    """
    ev_name: str
    p_setpoint: float

    def step(self, engine: "Engine") -> None:
        """Update the EV-facing setpoint from the engine state."""
        ...


class Engine:
    """
    Orchestrates the simulation run and coordinates components.

    The Engine advances simulated time, updates a shared :class:`WorldState`, and
    triggers all simulation components in a deterministic sequence: PCC →
    central controller → micro-controllers → chargers → EVs. It collects
    per-step metrics into :class:`StepRecord`s, converts those records to a
    wide ``pandas.DataFrame`` for analysis, and provides helpers to save and
    plot the results. The Engine also exposes utility methods for logging
    progress and for querying the number of active EVs at each timestep.

    Key responsibilities
    - Hold simulation configuration and component references.
    - Advance simulated time and enforce update ordering.
    - Maintain global world state (time, active EVs).
    - Coordinate inter-component interactions via the shared context.
    - Record, persist, and visualize per-step metrics.

    Attributes:
        config: Simulation configuration (:class:`SimConfig`) containing time
            step (`dt`), `time_horizon`, random `seed`, and other run
            parameters.
        priority_config: Priority allocation settings controlling which
            component calculates priorities, which strategy it uses, and which
            EVs participate.
        server: :class:`Server` instance used for component communication and
            delayed data exchange.
        pcc: PCC instance (implements :class:`PCCProtocol`) responsible for
            aggregating/observing total power and providing `step(engine)`.
        central_controller: Central controller (implements
            :class:`CentralControllerProtocol`) that computes global setpoints
            and exposes `p_setpoint` and `step(engine, t)`.
        micro_controllers: List of micro-controller instances (implementing
            :class:`MicroControllerProtocol`) that produce per-EV setpoints
            and expose `ev_name` and `p_setpoint`.
        mc_by_ev_name: Dict mapping EV name to its micro-controller instance
            for quick lookup.
        chargers: List of charger instances (implementing
            :class:`ChargerProtocol`) that translate micro-controller
            setpoints to EV-facing setpoints and expose `ev_name` and
            `p_setpoint`.
        charger_by_ev_name: Dict mapping EV name to its charger instance for
            quick lookup.
        evs: Dict of EV instances (keyed by EV name) implementing
            :class:`EVProtocol` and providing `step(engine, dt)`, `p_cons_kw`,
            and `next_p_cons_kw`.
        state: :class:`WorldState` tracking global simulation time (`t`) and
            number of active EVs (`n_active`).
        records: List of :class:`StepRecord` objects storing per-step metrics.
        records_df: Optional `pandas.DataFrame` produced by
            `convert_records_to_dataframe()` for analysis and plotting.
    """

    def __init__(
            self,
            config: SimConfig,
            priority_config: dict,
            server: Server,
            pcc: PCCProtocol,
            central_controller: CentralControllerProtocol,
            micro_controllers: list[MicroControllerProtocol],
            chargers: list[ChargerProtocol],
            evs: list[EVProtocol],
    ) -> None:
        self.config = config
        self.priority_config = priority_config
        self.server = server
        self.pcc = pcc
        self.central_controller = central_controller
        self.micro_controllers = micro_controllers
        self.mc_by_ev_name = {mc.ev_name: mc for mc in micro_controllers if mc.ev_name is not None}
        self.chargers = chargers
        self.charger_by_ev_name = {
            charger.ev_name: charger for charger in chargers if charger.ev_name is not None
        }
        self.evs: dict[str, EVProtocol] = {ev.name: ev for ev in evs}
        self.state = WorldState(t=0, n_active=0)
        self.records: list[StepRecord] = []
        self.records_df: pd.DataFrame | None = None

    def log_progress(self, t: float) -> None:
        """Log progress when *t* reaches a ten-percent boundary.

        Args:
            t: Current simulated time in seconds.
        """
        progress = (t / self.config.time_horizon) * 100
        # Log progress every 10% of the time horizon
        if progress % 10.0 < 1e-7:
            logger.info("Simulation progress: %.0f%%", progress)

    def update_world_state(self, t: float) -> None:
        """Update simulated time and count the EVs connected at that instant.

        Args:
            t: New simulated time in seconds.
        """
        self.state.t = t
        self.state.n_active = sum(
            1 for ev in self.evs.values() if ev.arrival_time <= t < ev.departure_time
        )

    def record_step(self) -> None:
        """Capture current aggregate and per-EV values in a step record.

        The new :class:`StepRecord` is appended to :attr:`records`.
        """
        record = StepRecord(
            t=self.state.t,
            p_total_kw=self.pcc.p_total_kw,
            p_central_setpoint_kw=self.central_controller.p_setpoint,
            n_active=self.state.n_active,
            cluster_cap_kw=0.0,  # Placeholder for cluster capacity in kW
            p_cons_kw={ev_name: ev.p_cons_kw for ev_name, ev in self.evs.items()},
            p_mc_setpoint_kw={ev_name: mc.p_setpoint for ev_name, mc in self.mc_by_ev_name.items()},
            p_charger_setpoint_kw={
                ev_name: charger.p_setpoint for ev_name, charger in self.charger_by_ev_name.items()
            },
            rel_priority={ev_name: mc.rel_priority_i for ev_name, mc in self.mc_by_ev_name.items()},
        )
        self.records.append(record)

    def run(self) -> None:
        """Run all simulation steps and convert the collected records.

        Results are stored in :attr:`records` and :attr:`records_df`.
        """
        logger.info("Starting simulation run.")
        for t in np.arange(self.state.t, self.config.time_horizon, self.config.dt):
            # update world state
            self.update_world_state(t)

            self.pcc.step(self)

            # central controller uses global view (single instance)
            self.central_controller.step(engine=self, t=self.state.t)

            # update all micro controllers (they read from central controller via server)
            for mc in self.micro_controllers:
                mc.step(self)

            # update all chargers (they read from micro controllers or server)
            for charger in self.chargers:
                charger.step(self)

            # update all EVs (each EV reads its corresponding charger via engine)
            for ev in self.evs.values():
                ev.step(self, self.config.dt)

            self.record_step()
            self.log_progress(t)

        logger.info("Simulation completed. Converting records to DataFrame.")
        self.convert_records_to_dataframe()

    def convert_records_to_dataframe(self) -> None:
        """Convert records to a wide, time-indexed results DataFrame.

        Per-EV dictionaries are expanded into prefixed columns. When a sample
        interval is configured, records are downsampled before conversion.
        The result is assigned to :attr:`records_df`.
        """
        records = self.records
        if self.config.sample_interval_s is not None:
            stride = max(1, round(self.config.sample_interval_s / self.config.dt))
            records = records[::stride]
        # Build dataframe from records and expand the per-EV power dict into columns
        df = pd.DataFrame([r.__dict__ for r in records])
        # Expand p_cons_kw dict (one column per EV) and prefix column names
        if "p_cons_kw" in df.columns:
            p_cons_df = pd.json_normalize(df["p_cons_kw"]).rename(
                columns=lambda x: f"p_cons_ev_{x}"
            )
            df = pd.concat([df.drop(columns=["p_cons_kw"]), p_cons_df], axis=1)
        # Expand p_mc_setpoint_kw dict (one column per EV) and prefix column names
        if "p_mc_setpoint_kw" in df.columns:
            p_mc_setpoint_df = pd.json_normalize(df["p_mc_setpoint_kw"]).rename(
                columns=lambda x: f"p_mc_setpoint_ev_{x}"
            )
            df = pd.concat([df.drop(columns=["p_mc_setpoint_kw"]), p_mc_setpoint_df], axis=1)
        # Expand p_charger_setpoint_kw dict (one column per EV) and prefix column names
        if "p_charger_setpoint_kw" in df.columns:
            p_charger_setpoint_df = pd.json_normalize(df["p_charger_setpoint_kw"]).rename(
                columns=lambda x: f"p_charger_setpoint_ev_{x}"
            )
            df = pd.concat(
                [df.drop(columns=["p_charger_setpoint_kw"]), p_charger_setpoint_df], axis=1
            )
        # Expand rel_priority dict (one column per EV) and prefix column names
        if "rel_priority" in df.columns:
            rel_priority_df = pd.json_normalize(df["rel_priority"]).rename(
                columns=lambda x: f"rel_priority_ev_{x}"
            )
            df = pd.concat([df.drop(columns=["rel_priority"]), rel_priority_df], axis=1)
        df.set_index("t", inplace=True)
        self.records_df = df

    def save_records_to_csv(self, filename: str) -> None:
        """Save converted records without overwriting an existing CSV file.

        Args:
            filename: Preferred output path. A numeric suffix is added when
                this path already exists.

        Raises:
            ValueError: If records have not been converted to a DataFrame.
        """
        if self.records_df is None:
            msg = "Records DataFrame is not created. Call convert_records_to_dataframe() first."
            raise ValueError(msg)
        path = Path(filename)
        stem = path.stem
        i = 1
        while path.exists():
            path = path.with_name(f"{stem}_{i}{path.suffix}")
            i += 1
        logger.info("Saving records to %s.", path)
        self.records_df.to_csv(path, index=True)

    def plot_records(
            self,
            save: bool = False,
            save_path: str = None,
            title: str = "Power Consumption and Setpoint",
    ) -> None:
        """Display the recorded power time series and optionally save it.

        Args:
            save: Whether to export the figure.
            save_path: Export path; defaults to a PDF in ``figures``.
            title: Figure title.

        Raises:
            ValueError: If records have not been converted to a DataFrame.
        """
        if self.records_df is None:
            msg = "Records DataFrame is not created. Call convert_records_to_dataframe() first."
            raise ValueError(msg)
        df = self.records_df
        max_time_s = df.index.max()
        if max_time_s > 2 * 60 * 60:
            time_scale = 60 * 60
            time_unit = "h"
        elif max_time_s > 3 * 60:
            time_scale = 60
            time_unit = "min"
        else:
            time_scale = 1
            time_unit = "s"
        time_index = df.index / time_scale
        fig = go.Figure()

        # total power consumption line
        if "p_total_kw" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=time_index,
                    y=df["p_total_kw"],
                    mode="lines",
                    name="P_cons",
                    line={"color": "blue"},
                )
            )

        # central controller setpoint as a step (post)
        if "p_central_setpoint_kw" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=time_index,
                    y=df["p_central_setpoint_kw"],
                    mode="lines",
                    name="P_ref",
                    line={"dash": "dash", "color": "black"},
                    line_shape="hv",
                )
            )

        # per-EV consumption lines (if present in dataframe)
        for ev in self.evs.values():
            col = f"p_cons_ev_{ev.name}"
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=time_index,
                        y=df[col],
                        mode="lines",
                        name=f"{ev.name}",
                        line={"dash": "dot"},
                    )
                )

        fig.update_layout(
            font={"family": "Times New Roman"},
            title={
                "text": title,
                "x": 0.5,
                "xanchor": "center",
                "font": {"family": "Times New Roman", "size": 40},
            },
            xaxis_title={"text": f"Time [{time_unit}]", "font": {"family": "Times New Roman", "size": 35}},
            yaxis_title={"text": "Power [kW]", "font": {"family": "Times New Roman", "size": 35}},
            xaxis={"tickfont": {"size": 25}},
            yaxis={"tickfont": {"size": 25}},
            legend={"font": {"family": "Times New Roman", "size": 35}},
            width=1600,
            height=600,
        )
        # if "p_central_setpoint_kw" in df.columns:
        y_max = df["p_central_setpoint_kw"].max() + max(5, df["p_central_setpoint_kw"].max() * 0.1)
        fig.update_yaxes(range=[-0.5, y_max])
        fig.show()
        if save:
            if save_path is None:
                save_path = "figures/power_consumption_plot.pdf"
            fig.write_image(save_path)
