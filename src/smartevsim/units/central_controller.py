"""Central charging-control strategies and their communication lifecycle."""

from abc import ABC, abstractmethod
from bisect import bisect_right

import gurobipy as gp

from smartevsim.engine import Engine
from smartevsim.utils.conversions import convert_A_to_kw
from smartevsim.utils.delay import Delay, UpdateDelay
from smartevsim.utils.priority import (
    calculate_even_priorities,
    calculate_total_urgency_priority,
    calculate_urgency_priorities,
    get_ev_names,
)


class CentralController(ABC):
    """Base class coordinating central calculations and delayed communication.

    The controller reads its required server inputs, updates a scheduled
    aggregate setpoint, computes controller-specific outputs, and advances
    calculation and publication delays at every simulation step.

    Attributes:
        name: Unique controller name used for server keys.
        p_setpoint: Current aggregate power reference in kilowatts.
        p_setpoint_schedule: Optional time-indexed reference schedule.
        update_interval: Optional gate controlling calculation frequency.
        set_delay: Optional output publication delay.
        get_delay: Optional input communication delay.
        calculation_delay: Optional computation delay.
        server_data: Most recently received input values.
        server_get_keys: Unit-variable keys required for calculations.
        output_data: Most recently completed controller outputs.
    """

    def __init__(
            self,
            name: str,
            init_p_setpoint: float | None = None,
            p_setpoint_schedule: dict | None = None,
            update_interval_s: float | None = None,
            set_delay_s: float | None = None,
            get_delay_s: float | None = None,
            calculation_delay_s: float | None = None,
            seed: int = 42,
    ) -> None:
        self.name = name
        self._init_p_setpoint(init_p_setpoint, p_setpoint_schedule)
        self.update_interval = UpdateDelay.create(update_interval_s, name, seed=seed)
        self.set_delay: Delay | None = Delay.create(set_delay_s, name)
        self.get_delay: Delay | None = Delay.create(get_delay_s, name)
        self.calculation_delay: Delay | None = Delay.create(calculation_delay_s, name)
        self.server_data: dict | None = None
        self.server_get_keys: list = []
        self.output_data: dict = {}

    def _init_p_setpoint(self, p_setpoint: float | None, p_setpoint_schedule: dict | None) -> None:
        if isinstance(p_setpoint, (int, float)):
            self.p_setpoint = float(p_setpoint)
            if p_setpoint_schedule is None:
                self.p_setpoint_schedule = None
        if isinstance(p_setpoint_schedule, dict):
            # sort the keys of the schedule
            p_setpoint_schedule = dict(sorted(p_setpoint_schedule.items()))
            self.p_setpoint_schedule = p_setpoint_schedule
            if p_setpoint is None:
                # initiliaze p_setpoint to the first (and earliest) value
                first_key = min(p_setpoint_schedule.keys())
                self.p_setpoint = p_setpoint_schedule[first_key]
        if (
            not isinstance(self.p_setpoint, (int, float)) 
            and not isinstance(self.p_setpoint_schedule, dict)
        ):
            msg = (
                f"Invalid combination of init_p_setpoint and p_setpoint_schedule types: "
                f"{type(p_setpoint).__name__} and {type(p_setpoint_schedule).__name__}. Specify "
                f"init_p_setpoint as a float and/or p_setpoint_schedule as a dict with time as "
                f"keys and corresponding setpoints as values."
            )
            raise TypeError(msg)

    def update_p_setpoint(self, t: float) -> None:
        """Apply the latest scheduled aggregate setpoint at time *t*.

        Args:
            t: Current simulated time in seconds.
        """
        if self.p_setpoint_schedule is not None:
            # update p_setpoint to the value corresponding to the largest key <= t
            t_keys = list(self.p_setpoint_schedule.keys())
            idx = bisect_right(t_keys, t) - 1
            if idx >= 0:
                self.p_setpoint = self.p_setpoint_schedule[t_keys[idx]]

    def update_server_get_keys(self, engine: Engine) -> None:
        """
        Method to update the server keys if for instance the server data depends on the
        currently connected EVs. Called before each calculation to ensure keys are up-to-date.
        Optional hook which subclasses may override.

        Args:
            engine: Engine used to determine the required dynamic inputs.
        """
        return None

    def get_server_data(self, engine: Engine) -> None:
        """Read current inputs from the server, respecting the read delay.

        Args:
            engine: Engine providing simulated time and the shared server.
        """
        self.update_server_get_keys(engine)
        server_data = engine.server.get_from_keys(server_keys=self.server_get_keys)
        if self.get_delay is None:
            self.server_data = server_data
        else:
            self.get_delay.schedule(engine.state.t, server_data)
            delayed_data = self.get_delay.pop_expired(engine.state.t)
            if delayed_data is not None:
                self.server_data = delayed_data

    def set_server_data(self, engine: Engine) -> None:
        """Publish controller output immediately or through the set delay.

        Args:
            engine: Engine providing simulated time and the shared server.
        """
        server_data = self.create_server_data()
        if self.set_delay is None:
            engine.server.set_from_keys(server_data=server_data)
        else:
            self.set_delay.schedule(engine.state.t, server_data)

    def apply_calculation_delay(self, engine: Engine, t: float, output_data: dict) -> None:
        """Apply calculated output immediately or schedule delayed completion.

        Args:
            engine: Engine providing the shared server.
            t: Current simulated time in seconds.
            output_data: Newly calculated controller outputs.
        """
        if self.calculation_delay is None:
            self.output_data = output_data
            self.set_server_data(engine)
        else:
            self.calculation_delay.schedule(t, output_data)

    def step_calculation_delay(self, engine: Engine, t: float) -> None:
        """Apply the latest calculation result due by time *t*.

        Args:
            engine: Engine providing the shared server.
            t: Current simulated time in seconds.
        """
        if self.calculation_delay is not None:
            # Ensure any remaining delayed data is applied, even if no new data is scheduled.
            delayed_data = self.calculation_delay.pop_expired(t)
            if delayed_data is not None:
                self.output_data = delayed_data
                self.set_server_data(engine)

    def step_set_delay(self, engine: Engine) -> None:
        """Publish controller data whose set delay has expired.

        Args:
            engine: Engine providing simulated time and the shared server.
        """
        if self.set_delay is not None:
            delayed_data = self.set_delay.pop_expired(engine.state.t)
            if delayed_data is not None:
                engine.server.set_from_keys(server_data=delayed_data)

    def has_required_inputs(self) -> bool:
        """Return whether every required server input is available.

        Returns:
            ``True`` when server data exists and contains no missing values.
        """
        return self.server_data is not None and None not in self.server_data.values()

    def calculate_priorities(self, engine: Engine, t: float) -> dict | float:
        """Calculate priorities according to the configured method and owner.

        Args:
            engine: Engine containing priority configuration and EV state.
            t: Current simulated time in seconds.

        Returns:
            Per-EV priorities when centrally calculated, otherwise the total
            urgency scalar needed by micro-controllers.

        Raises:
            ValueError: If the configured strategy or owner is invalid.
        """
        if engine.priority_config.priority_method not in {"even", "urgency"}:
            msg = (
                f"Invalid priority method: {engine.priority_config.priority_method}. Expected "
                f"'even' or 'urgency'."
            )
            raise ValueError(msg)
        if engine.priority_config.responsible_controller == "central":
            if engine.priority_config.priority_method == "even":
                priorities = calculate_even_priorities(engine)
            elif engine.priority_config.priority_method == "urgency":
                priorities = calculate_urgency_priorities(engine, t)
        elif engine.priority_config.responsible_controller == "micro":
            # If microcontrollers are responsible for calculating priorities, central controller
            # can still calculate total priority for urgency method to enable priority-proportional
            # control.
            if engine.priority_config.priority_method == "urgency":
                priorities = calculate_total_urgency_priority(engine, t)
            elif engine.priority_config.priority_method == "even":
                priorities = 0.0
        else:
            msg = (
                f"Invalid priority responsibility configuration: "
                f"{engine.priority_config.responsible_controller}. "
                f"Expected 'central' or 'micro'."
            )
            raise ValueError(msg)

        return priorities

    def step(self, engine: Engine, t: float | None = None) -> None:
        """Advance controller calculation and communication by one step.

        Args:
            engine: Engine providing timing, EV, and server state.
            t: Optional calculation time forwarded to controller strategies.
        """
        # Always advance read-delay state each step so delayed data reception is
        # decoupled from the controller update interval.
        self.get_server_data(engine)

        if (
            (self.update_interval is None or self.update_interval.should_update(engine.state.t))
            and self.has_required_inputs()
        ):
            self.update_p_setpoint(engine.state.t)
            output_data = self.calculate_outputs(engine, t)
            self.apply_calculation_delay(engine, engine.state.t, output_data)

        self.step_calculation_delay(engine, engine.state.t)
        self.step_set_delay(engine)

    @abstractmethod
    def calculate_outputs(self, engine: Engine, t: float | None = None) -> dict:
        """Calculate controller-specific output values.

        Args:
            engine: Engine providing simulation state.
            t: Optional calculation time in seconds.

        Returns:
            Controller-specific output mapping.
        """
        msg = "Subclasses of CentralController must implement method calculate_outputs."
        raise NotImplementedError(msg)

    @abstractmethod
    def create_server_data(self) -> dict[tuple[str, str], float]:
        """Convert the current output values to server-keyed data.

        Returns:
            Output values keyed by ``(unit, variable)`` tuples.
        """
        msg = "Subclasses of CentralController must implement method create_server_data."
        raise NotImplementedError(msg)


class ConstantCentralController(CentralController):
    """Publish the aggregate setpoint and calculated EV priorities.

    Attributes:
        priorities: Most recently calculated per-EV priorities or total
            priority scalar.
    """
    def __init__(
            self,
            name: str,
            init_p_setpoint: float | None = None,
            p_setpoint_schedule: dict | None = None,
            update_interval_s: float | None = None,
            set_delay_s: float | None = None,
            get_delay_s: float | None = None,
            calculation_delay_s: float | None = None,
            seed: int = 42,
    ) -> None:
        super().__init__(
            name=name,
            init_p_setpoint=init_p_setpoint,
            p_setpoint_schedule=p_setpoint_schedule,
            update_interval_s=update_interval_s,
            set_delay_s=set_delay_s,
            get_delay_s=get_delay_s,
            calculation_delay_s=calculation_delay_s,
            seed=seed,
        )
        self.priorities: dict[str, float] = {}
        self.server_get_keys: list[tuple[str, str]] = []

    def calculate_outputs(self, engine: Engine, t: float) -> dict:
        """Return the current reference and priorities for time *t*.

        Args:
            engine: Engine providing EV and priority state.
            t: Calculation time in seconds.

        Returns:
            Mapping containing ``p_setpoint`` and ``priorities``.
        """
        priorities = self.calculate_priorities(engine, t)
        output_data = {
            "p_setpoint": self.p_setpoint,
            "priorities": priorities,
        }
        return output_data

    def create_server_data(self) -> dict[tuple[str, str], float]:
        """Return validated reference and priorities keyed for publication.

        Returns:
            Current outputs keyed by controller name and variable.
        """
        p_setpoint = self.output_data.get("p_setpoint")
        priorities = self.output_data.get("priorities")
        if not isinstance(p_setpoint, (int, float)):
            msg = f"Invalid p_setpoint output type: {type(p_setpoint).__name__}"
            raise TypeError(msg)
        if not isinstance(priorities, (dict, float)):
            msg = f"Invalid priorities output type: {type(priorities).__name__}"
            raise TypeError(msg)
        
        server_data = {
            (self.name, "p_setpoint"): float(p_setpoint),
            (self.name, "priorities"): priorities,
        }
        return server_data


class PIConstantCentralController(ConstantCentralController):
    """Adjust a constant reference with bounded proportional-integral control.

    Attributes:
        k_p: Proportional gain.
        k_i: Integral gain per update.
        I_max: Absolute integral-term bound.
        I: Current integral term.
        n_evs: EV count observed during the previous update.
    """
    """
    Baseline: always return the same action, regardless of state.
    """
    def __init__(
            self,
            name: str,
            init_p_setpoint: float | None = None,
            p_setpoint_schedule: dict | None = None,
            update_interval_s: float | None = None,
            set_delay_s: float | None = None,
            get_delay_s: float | None = None,
            calculation_delay_s: float | None = None,
            seed: int = 42,
            k_p: float = 0.1,
            k_i: float = 0.1,
            I_max: float = 5.0,
    ) -> None:
        super().__init__(
            name=name,
            init_p_setpoint=init_p_setpoint,
            p_setpoint_schedule=p_setpoint_schedule,
            update_interval_s=update_interval_s,
            set_delay_s=set_delay_s,
            get_delay_s=get_delay_s,
            calculation_delay_s=calculation_delay_s,
            seed=seed,
        )
        self.k_p: float = k_p
        self.k_i: float = k_i
        self.I_max: float = I_max
        self.I = 0.0
        self.n_evs: int = 0
        self.server_get_keys: list[tuple[str, str]] = [
            ("pcc", "p_total_kw"),
        ]

    def update_integral(self, e: float, n_evs: int) -> None:
        """Update and bound the integral term, resetting after EV arrivals.

        Args:
            e: Aggregate tracking error in kilowatts.
            n_evs: Current number of prioritized EVs.
        """
        self.I += self.k_i * e
        self.I = max(-self.I_max, min(self.I_max, self.I))
        if n_evs > self.n_evs:
            # Reset integral term when new EVs arrive, to avoid overcompensation based on old error.
            self.I = 0.0
        self.n_evs = n_evs

    def calculate_setpoint(self, n_evs: int) -> float:
        """Calculate the PI-compensated aggregate setpoint.

        Args:
            n_evs: Current number of prioritized EVs.

        Returns:
            Compensated aggregate setpoint in kilowatts.
        """
        p_total_kw = self.server_data.get(("pcc", "p_total_kw"))
        if not isinstance(p_total_kw, (int, float)):
            p_total_kw = 0.0
        p_ref = self.p_setpoint
        e = p_ref - p_total_kw
        self.update_integral(e, n_evs)
        p_setpoint_i = self.p_setpoint + self.k_p * e + self.I
        return p_setpoint_i

    def calculate_outputs(self, engine: Engine, t: float) -> dict:
        """Return the PI-compensated reference and priorities for time *t*.

        Args:
            engine: Engine providing aggregate measurement and priority state.
            t: Calculation time in seconds.

        Returns:
            Mapping containing the compensated setpoint and priorities.
        """
        priorities = self.calculate_priorities(engine, t)
        n_evs = len(priorities) if isinstance(priorities, dict) else 0
        p_setpoint = self.calculate_setpoint(n_evs)
        output_data = {
            "p_setpoint": p_setpoint,
            "priorities": priorities,
        }
        return output_data


class OptimalCentralController(CentralController):
    """Allocate integer-current EV setpoints through centralized optimization.

    Attributes:
        i_min_a: Minimum optimized charging current in amperes.
        priorities: Priorities used by the current optimization problem.
    """

    def __init__(
            self,
            name: str,
            init_p_setpoint: float | None = None,
            p_setpoint_schedule: dict | None = None,
            update_interval_s: float | None = None,
            set_delay_s: float | None = None,
            get_delay_s: float | None = None,
            calculation_delay_s: float | None = None,
            seed: int = 42,
            i_min_a: float = 6.0,
    ) -> None:
        super().__init__(
            name=name,
            init_p_setpoint=init_p_setpoint,
            p_setpoint_schedule=p_setpoint_schedule,
            update_interval_s=update_interval_s,
            set_delay_s=set_delay_s,
            get_delay_s=get_delay_s,
            calculation_delay_s=calculation_delay_s,
            seed=seed,
        )
        self.i_min_a: float = i_min_a
        self.server_get_keys: list[tuple[str, str]] = []
        self.priorities: dict[str, float] = {}

    def update_server_get_keys(self, engine: Engine) -> None:
        """Update the server keys to get the max power of currently connected EVs."""
        ev_names = get_ev_names(engine)
        self.server_get_keys = [(ev_name, "p_max_kw") for ev_name in ev_names]

    def calculate_setpoint(self, engine: Engine) -> dict:
        """Optimize per-EV integer-current setpoints.

        Args:
            engine: Engine providing the current EV population.

        Returns:
            Mapping from prioritized EV names to power setpoints in kilowatts.
        """
        # ev_names = get_connected_ev_names(engine, engine.state.t)
        ev_names = [ev for ev in self.priorities]
        if not ev_names:
            return {}
        default_p_max = convert_A_to_kw(16.0)
        p_max = {ev: self.server_data.get((ev, "p_max_kw"), default_p_max) for ev in ev_names}

        # create optimization problem
        opt_prob = gp.Model("central_controller_opt")
        opt_prob.Params.OutputFlag = 0

        # decision variables
        i_setpoint = opt_prob.addVars(
            ev_names, name="i_setpoint", lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.INTEGER
        )
        # auxilliary variable 
        p_setpoint = {ev: convert_A_to_kw(i_setpoint[ev]) for ev in ev_names}

        # constraints
        opt_prob.addConstr(gp.quicksum(p_setpoint[ev] for ev in ev_names) <= self.p_setpoint)
        opt_prob.addConstrs((p_setpoint[ev] <= p_max[ev] for ev in ev_names), name="p_max")
        opt_prob.addConstrs((self.i_min_a <= i_setpoint[ev] for ev in ev_names), name="i_min")

        # objective: maximize total setpoint across all EVs
        total_consumption = gp.quicksum(p_setpoint[ev] for ev in ev_names)
        avg_consumption = total_consumption / len(ev_names)

        # consumption-priority-ratio = ev setpoint / (ev priority * self.p_setpoint)
        cpr_list = [p_setpoint[ev] / (self.priorities[ev] * self.p_setpoint) for ev in ev_names]
        cpr_mean = gp.quicksum(cpr_list) / len(ev_names)
        cpr_variance = gp.quicksum((cpr - cpr_mean)**2 for cpr in cpr_list) / len(ev_names)
        opt_prob.setObjective(avg_consumption - cpr_variance, gp.GRB.MAXIMIZE)

        opt_prob.optimize()

        # add 1e-3 to avoid rounding further down due to floating point precision issues
        sol_p_setpoint_i = {ev: convert_A_to_kw(i_setpoint[ev].X + 1e-3) for ev in ev_names}

        return sol_p_setpoint_i

    def calculate_outputs(self, engine: Engine, t: float) -> dict:
        """Calculate priorities and optimized per-EV setpoints for time *t*.

        Args:
            engine: Engine providing EV and priority state.
            t: Calculation time in seconds.

        Returns:
            Mapping containing optimized per-EV setpoints.
        """
        self.priorities = self.calculate_priorities(engine, t)
        p_setpoint = self.calculate_setpoint(engine)
        output_data = {"p_setpoint": p_setpoint}
        return output_data

    def create_server_data(self) -> dict[tuple[str, str], float]:
        """Return validated optimized setpoints keyed for publication.

        Returns:
            Optimized setpoint mapping keyed by controller name.
        """
        p_setpoint = self.output_data.get("p_setpoint")
        if not isinstance(p_setpoint, dict):
            msg = f"Invalid p_setpoint output type: {type(p_setpoint).__name__}"
            raise TypeError(msg)
        server_data = {(self.name, "p_setpoint"): p_setpoint}
        return server_data
