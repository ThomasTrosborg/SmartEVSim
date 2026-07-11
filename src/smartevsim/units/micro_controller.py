"""Per-EV charging controllers and silent-consensus variants."""

import logging
import math
from abc import ABC, abstractmethod

import numpy as np

from smartevsim.engine import Engine
from smartevsim.utils.conversions import convert_A_to_kw, convert_kw_to_A
from smartevsim.utils.delay import Delay, UpdateDelay
from smartevsim.utils.priority import calculate_even_priority, calculate_urgency_priority

logger = logging.getLogger(__name__)


class MicroController(ABC):
    """Base class for per-EV control and delayed server communication.

    Attributes:
        name: Unique controller name used for server keys.
        ev_name: Name of the controlled EV, if assigned.
        p_setpoint: Current EV power setpoint in kilowatts.
        rel_priority_i: Current relative priority of the associated EV.
        i_rounding_method: Current rounding mode.
        i_granularity: Allowed current increment in amperes.
        update_interval: Optional gate controlling calculation frequency.
        set_delay: Optional output publication delay.
        get_delay: Optional input communication delay.
        calculation_delay: Optional computation delay.
        server_data: Most recently received server values.
        server_get_keys: Unit-variable keys required for calculations.
    """

    def __init__(
            self,
            name: str,
            ev_name: str,
            init_p_setpoint: float = 0.0,
            update_interval_s: float | None = None,
            set_delay_s: float | None = None,
            get_delay_s: float | None = None,
            calculation_delay_s: float | None = None,
            i_min: float = 6.0,
            i_rounding_method: str = "floor",
            i_granularity: float = 1.0,
            seed: int = 42,
    ) -> None:
        self.name = name
        self.ev_name = ev_name
        self.p_setpoint = init_p_setpoint
        self.i_min = i_min
        self.i_rounding_method = i_rounding_method
        self.i_granularity = i_granularity
        self.rel_priority_i: float | None = None
        self.update_interval = UpdateDelay.create(update_interval_s, name, seed=seed)
        self.set_delay: Delay | None = Delay.create(set_delay_s, name)
        self.get_delay: Delay | None = Delay.create(get_delay_s, name)
        self.calculation_delay: Delay | None = Delay.create(calculation_delay_s, name)
        self.server_data: dict[tuple[str, str], float] | None = None
        self.server_get_keys: list = [(ev_name, "arrival_time"), (ev_name, "departure_time")]

    def get_server_data(self, engine: Engine) -> None:
        """Read required server inputs, respecting the read delay.

        Args:
            engine: Engine providing simulated time and the shared server.
        """
        server_data = engine.server.get_from_keys(server_keys=self.server_get_keys)
        if self.get_delay is None:
            self.server_data = server_data
        else:
            self.get_delay.schedule(engine.state.t, server_data)
            delayed_data = self.get_delay.pop_expired(engine.state.t)
            if delayed_data is not None:
                self.server_data = delayed_data

    def set_server_data(self, engine: Engine) -> None:
        """Publish the current setpoint immediately or after a set delay.

        Args:
            engine: Engine providing simulated time and the shared server.
        """
        server_data = {(self.name, "p_setpoint"): float(self.p_setpoint)}
        if self.set_delay is None:
            engine.server.set_from_keys(server_data=server_data)
        else:
            self.set_delay.schedule(engine.state.t, server_data)

    def has_required_inputs(self) -> bool:
        """Return whether every required server input is available.

        Returns:
            ``True`` when server data exists and contains no missing values.
        """
        return self.server_data is not None and None not in self.server_data.values()

    def is_connected(self, t: float) -> bool:
        """Return whether the associated EV is connected at time *t*.

        Args:
            t: Simulated time in seconds.

        Returns:
            ``True`` when ``t`` lies in the EV's connection interval.
        """
        if self.ev_name is None:
            return False
        arrival_time = self.server_data[(self.ev_name, "arrival_time")]
        departure_time = self.server_data[(self.ev_name, "departure_time")]
        return arrival_time <= t < departure_time

    def round_to_integer_current_setpoint(self, p_setpoint_kw: float) -> float:
        """Round a power setpoint through configured current constraints.

        Args:
            p_setpoint_kw: Unconstrained power setpoint in kilowatts.

        Returns:
            Power setpoint corresponding to the rounded current.

        Raises:
            ValueError: If the configured rounding method is unsupported.
        """
        """
        Round the current setpoint according to the specified rounding method and granularity, and
        apply the minimum current constraint.
        """
        i_setpoint = convert_kw_to_A(p_setpoint_kw)
        if self.i_rounding_method == "round":
            i_setpoint = round(i_setpoint / self.i_granularity, 0) * self.i_granularity
        elif self.i_rounding_method == "floor":
            i_setpoint = math.floor(i_setpoint / self.i_granularity) * self.i_granularity
        elif self.i_rounding_method == "ceil":
            i_setpoint = math.ceil(i_setpoint / self.i_granularity) * self.i_granularity
        else:
            msg = (
                f"Invalid current rounding method in micro controller {self.name}: "
                f"{self.i_rounding_method}. Expected 'round', 'floor' or 'ceil'."
            )
            raise ValueError(msg)
        if i_setpoint > 0:
            i_setpoint = max(i_setpoint, self.i_min)
        # add a small delta to avoid further rounding in the charger due to floating point precision
        i_setpoint += 1e-3
        p_setpoint_kw = convert_A_to_kw(i_setpoint)
        return p_setpoint_kw

    def apply_calculation_delay(self, engine: Engine, t: float, p_setpoint: float) -> None:
        """Apply a result immediately or queue delayed calculation completion.

        Args:
            engine: Engine providing the shared server.
            t: Current simulated time in seconds.
            p_setpoint: Newly calculated power setpoint in kilowatts.
        """
        if self.calculation_delay is None:
            self.p_setpoint = p_setpoint
            self.set_server_data(engine)
        else:
            self.calculation_delay.schedule(t, p_setpoint)

    def step_calculation_delay(self, engine: Engine, t: float) -> None:
        """Apply the latest controller result due by time *t*.

        Args:
            engine: Engine providing the shared server.
            t: Current simulated time in seconds.
        """
        if self.calculation_delay is not None:
            delayed_setpoint = self.calculation_delay.pop_expired(t)
            if delayed_setpoint is not None:
                self.p_setpoint = delayed_setpoint
                self.set_server_data(engine)

    def step_set_delay(self, engine: Engine) -> None:
        """Publish controller output whose set delay has expired.

        Args:
            engine: Engine providing simulated time and the shared server.
        """
        if self.set_delay is not None:
            delayed_data = self.set_delay.pop_expired(engine.state.t)
            if delayed_data is not None:
                engine.server.set_from_keys(server_data=delayed_data)

    def set_relative_priority(self, engine: Engine) -> None:
        """Update this EV's relative priority from central or local data.

        Args:
            engine: Engine containing priority configuration and EV state.
        """
        if engine.priority_config.responsible_controller == "central":
            priorities = self.server_data.get((engine.central_controller.name, "priorities"))
            if isinstance(priorities, dict):
                self.rel_priority_i = priorities.get(self.ev_name)
            else:
                raise ValueError("Invalid priority type for central priority calculation.")
        elif engine.priority_config.responsible_controller == "micro":
            if engine.priority_config.priority_method == "even":
                self.rel_priority_i = calculate_even_priority(engine)
            elif engine.priority_config.priority_method == "urgency":
                total_priority = self.server_data.get(
                    (engine.central_controller.name, "total_priority")
                )
                self.rel_priority_i = calculate_urgency_priority(
                    ev_name=self.ev_name,
                    engine=engine,
                    total_priority=total_priority,
                )
        else:
            raise ValueError("Invalid responsible_controller in priority_config.")

    @abstractmethod
    def calculate_setpoint(self, engine: Engine) -> float:
        """Calculate the next power setpoint for the associated EV.

        Args:
            engine: Engine providing simulation and server state.

        Returns:
            Requested EV power in kilowatts.
        """
        msg = "Subclasses of MicroController must implement abstract method calculate_setpoint."
        raise NotImplementedError(msg)

    def step(self, engine: Engine) -> None:
        """Advance controller calculation and communication by one step.

        Args:
            engine: Engine providing timing, EV, and server state.
        """
        """Update micro controller setpoint from the Server."""
        # Always advance read-delay state each step so delayed data reception is
        # decoupled from the controller update interval.
        self.get_server_data(engine)

        if not self.has_required_inputs():
            pass
        elif not self.is_connected(engine.state.t):
            # if no EV is connected, set the setpoint to 0 and clear the calculation delay queue
            self.p_setpoint = 0.0
            if self.calculation_delay is not None:
                self.calculation_delay.clear()
            self.set_server_data(engine)
        elif self.update_interval is None or self.update_interval.should_update(engine.state.t):
            self.set_relative_priority(engine)
            p_setpoint = self.calculate_setpoint(engine)
            self.apply_calculation_delay(engine, engine.state.t, p_setpoint)

        self.step_calculation_delay(engine, engine.state.t)
        self.step_set_delay(engine)


class RuleMicroController(MicroController):
    """Choose charging actions from aggregate and individual rule states.

    Attributes:
        policy: Action lookup indexed by aggregate and individual states.
        epsilon: Tolerance used when classifying control states.
    """

    def __init__(
        self,
        name: str,
        ev_name: str,
        init_p_setpoint: float = 0.0,
        update_interval_s: float | None = None,
        set_delay_s: float | None = None,
        get_delay_s: float | None = None,
        calculation_delay_s: float | None = None,
        i_min: float = 6.0,
        i_rounding_method: str = "round",
        i_granularity: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__(
            name,
            ev_name,
            init_p_setpoint,
            update_interval_s,
            set_delay_s,
            get_delay_s,
            calculation_delay_s,
            i_min,
            i_rounding_method,
            i_granularity,
            seed,
        )
        self.server_get_keys = [
            (ev_name, "arrival_time"),
            (ev_name, "departure_time"),
            ("cc", "priorities"),
            ("cc", "p_setpoint"),
            ("pcc", "p_total_kw"),
        ]
        self.policy: np.ndarray = np.array([[1, 1, 1], [1, 0, -1], [1, 0, -1]])
        self.policy = self.policy * convert_A_to_kw(max(self.i_granularity, 0.5))
        self.epsilon = convert_A_to_kw(max(self.i_granularity, 0.5))

    def has_required_inputs(self) -> bool:
        """Return whether this variant's required values and priority are available."""
        if not super().has_required_inputs():
            return False
        # Additionally check that priorities contains the key for this micro controller's EV
        priorities = self.server_data.get(("cc", "priorities"))
        if isinstance(priorities, dict) and self.ev_name in priorities:
            return True
        if isinstance(priorities, (float, int)):
            return True
        # raise an error if priorities is neither None, dict or float
        if priorities is not None and not isinstance(priorities, (dict, float)):
            msg = (
                f"Invalid priorities type for micro controller {self.name}: {type(priorities)}. "
                f"Expected dict with relative priorities or float with total priority."
            )
            raise ValueError(msg)
        return False

    def get_aggregate_state(self, p_ref: float, p_total_kw: float) -> int:
        """Classify aggregate consumption relative to its reference.

        Args:
            p_ref: Aggregate reference in kilowatts.
            p_total_kw: Measured aggregate consumption in kilowatts.

        Returns:
            Discrete aggregate state used by the rule policy.
        """
        if p_total_kw < p_ref - self.epsilon:
            return 0
        elif p_total_kw > p_ref:
            return 2
        else:
            return 1

    def get_individual_state(self, p_cons_kw_i: float, p_ref: float) -> int:
        """Classify individual consumption relative to its priority share.

        Args:
            p_cons_kw_i: Associated EV consumption in kilowatts.
            p_ref: Aggregate reference in kilowatts.

        Returns:
            Discrete individual state used by the rule policy.
        """
        if p_cons_kw_i < self.rel_priority_i * p_ref - self.epsilon:
            return 0
        elif p_cons_kw_i > self.rel_priority_i * p_ref:
            return 2
        else:
            return 1

    def get_action(
            self,
            aggregate_state: int,
            individual_state: int,
            p_cons_kw_i: float,
            p_ref: float
    ) -> float:
        """Look up the policy action for aggregate and individual states.

        Args:
            aggregate_state: Discrete aggregate state.
            individual_state: Discrete EV state.
            p_cons_kw_i: Current EV consumption in kilowatts.
            p_ref: Aggregate reference in kilowatts.

        Returns:
            Next requested EV power in kilowatts.
        """
        return self.policy[aggregate_state, individual_state]

    def calculate_setpoint(self, engine: Engine) -> float:
        """Calculate a rule-policy EV power request.

        Args:
            engine: Engine providing aggregate, EV, and server state.

        Returns:
            Rounded requested EV power in kilowatts.
        """
        p_ref = self.server_data[(engine.central_controller.name, "p_setpoint")]
        p_total_kw = self.server_data[(engine.pcc.name, "p_total_kw")]
        p_cons_kw_i = engine.evs[self.ev_name].next_p_cons_kw
        aggregate_state = self.get_aggregate_state(p_ref, p_total_kw)
        individual_state = self.get_individual_state(p_cons_kw_i, p_ref)
        p_setpoint_delta = self.get_action(
            aggregate_state, individual_state, p_cons_kw_i, p_ref
        )
        p_setpoint_i = self.p_setpoint + p_setpoint_delta
        # setpoint cannot be lower than the setpoint corresponding to its priority
        p_setpoint_i = max(p_setpoint_i, self.rel_priority_i * p_ref)
        # setpoint cannot exceed 11.1 kW (corresponding to 16 A)
        p_setpoint_i = min(p_setpoint_i, 11.1)
        p_setpoint_i_rounded = self.round_to_integer_current_setpoint(p_setpoint_i)
        return p_setpoint_i_rounded


class CentralizedMicroController(MicroController):
    """Read per-EV setpoints produced by the centralized controller.

    The class adds no persistent public attributes beyond
    :class:`MicroController`.
    """

    def __init__(
        self,
        name: str,
        ev_name: str,
        init_p_setpoint: float = 0.0,
        update_interval_s: float | None = None,
        set_delay_s: float | None = None,
        get_delay_s: float | None = None,
        calculation_delay_s: float | None = None,
        i_min: float = 6.0,
        i_rounding_method: str = "round",
        i_granularity: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__(
            name,
            ev_name,
            init_p_setpoint,
            update_interval_s,
            set_delay_s,
            get_delay_s,
            calculation_delay_s,
            i_min,
            i_rounding_method,
            i_granularity,
            seed,
        )
        self.server_get_keys: list = [
            (ev_name, "arrival_time"),
            (ev_name, "departure_time"),
            ('cc', "p_setpoint"),
        ]

    def has_required_inputs(self) -> bool:
        """Return whether the centralized per-EV setpoint is available."""
        if not super().has_required_inputs():
            return False
        # Additionally check that p_setpoint contains the key for this micro controller's EV
        p_setpoint = self.server_data.get(("cc", "p_setpoint"))
        if isinstance(p_setpoint, dict) and self.ev_name in p_setpoint:
            return True
        # raise an error if p_setpoint is neither None, dict or float
        if p_setpoint is not None and not isinstance(p_setpoint, dict):
            msg = (
                f"Invalid p_setpoint type for centralized micro controller {self.name}: "
                f"{type(p_setpoint)}. Expected dict with p_setpoints for each EV."
            )
            raise ValueError(msg)
        return False

    def set_relative_priority(self, engine: Engine) -> None:
        """Copy this EV's priority from the central controller.

        Args:
            engine: Engine containing the central controller.
        """
        self.rel_priority_i = engine.central_controller.priorities.get(self.ev_name)

    def calculate_setpoint(self, engine: Engine) -> float:
        """Read this EV's centrally optimized power request.

        Args:
            engine: Engine identifying the central-controller server key.

        Returns:
            Associated EV power setpoint in kilowatts.
        """
        p_setpoint = self.server_data[(engine.central_controller.name, "p_setpoint")]
        return p_setpoint[self.ev_name]

