"""Shared configuration, state, and result data structures."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimConfig:
    """Immutable timing, random-seed, and result-sampling configuration.

    Attributes:
        dt: Simulation step duration in seconds.
        time_horizon: Total simulated duration in seconds.
        seed: Root seed used to derive component random-number generators.
        sample_interval_s: Optional interval for downsampling recorded results.
    """
    dt: float
    time_horizon: float
    seed: int
    sample_interval_s: float | None = None

@dataclass(frozen=True)
class PriorityConfig:
    """Immutable configuration for assigning control priority among EVs.

    Attributes:
        responsible_controller: Component responsible for priorities;
            ``"central"`` or ``"micro"``.
        priority_method: Allocation strategy; ``"urgency"`` or ``"even"``.
        evs: Participating population: ``"all"``, ``"connected"``, or a fixed
            EV count.
    """
    responsible_controller: str # "central" or "micro"
    priority_method: str # "urgency" or "equal"
    evs: str | int = "all"  # "all", "connected", or number of EVs to use

@dataclass
class WorldState:
    """Mutable simulation-wide time and connection state.

    Attributes:
        t: Current simulated time in seconds.
        n_active: Number of currently connected EVs.
    """
    t: float
    n_active: int

@dataclass
class StepRecord:
    """Aggregate and per-EV measurements captured at one simulation step.

    Attributes:
        t: Simulated time in seconds.
        p_total_kw: Aggregate EV power consumption.
        p_central_setpoint_kw: Aggregate central-controller reference.
        n_active: Number of connected EVs.
        cluster_cap_kw: Available cluster capacity.
        p_cons_kw: Per-EV consumption.
        p_mc_setpoint_kw: Per-EV micro-controller setpoints.
        p_charger_setpoint_kw: Per-EV charger setpoints.
        rel_priority: Per-EV relative priorities.
    """
    t: float
    p_total_kw: float
    p_central_setpoint_kw: float
    n_active: int
    cluster_cap_kw: float
    p_cons_kw: dict[str, float] = field(default_factory=dict)
    p_mc_setpoint_kw: dict[str, float] = field(default_factory=dict)
    p_charger_setpoint_kw: dict[str, float] = field(default_factory=dict)
    rel_priority: dict[str, float] = field(default_factory=dict)
