"""Electric-vehicle charging dynamics and communication behavior."""

from bisect import bisect_right

from smartevsim.engine import Engine
from smartevsim.utils.conversions import convert_A_to_kw
from smartevsim.utils.delay import Delay


class EV:
    """Represent an electric vehicle and its charging dynamics.

    Attributes:
        name: Unique name of the EV.
        arrival_time: Connection start time in seconds.
        departure_time: Connection end time in seconds.
        e_required_kwh: Energy requested during the connection in kWh.
        soc: Current state of charge as a fraction of battery capacity.
        p_max_kw: Maximum charging power in kilowatts.
        battery_capacity_kwh: Usable battery capacity in kWh.
        ramp_up_kw_pr_s: Optional ramp-up limit in kW per second.
        ramp_down_kw_pr_s: Optional ramp-down limit in kW per second.
        efficiency: Constant or power-indexed charging efficiency.
        actuation_delay: Optional delay for applying ordinary setpoints.
        wake_up_delay_s: Optional delay for transitioning from zero power.
        set_delay: Optional delay for publishing EV state.
        get_delay: Optional server read delay.
        applied_p_setpoint: Most recently actuated charger setpoint in kW.
        e_charged_kwh: Energy delivered during the simulated connection.
        p_cons_kw: Power consumption in the current step.
        next_p_cons_kw: Power consumption calculated for the next step.
    """

    def __init__(
            self,
            server, # type is Server, but won't specify to avoid circular import
            name: str,
            arrival_time: float,
            departure_time: float,
            e_required_kwh: float,
            init_soc: float,
            p_max_kw: float,
            battery_capacity_kwh: float,
            ramp_up_kw_pr_s: float | None = None,
            ramp_down_kw_pr_s: float | None = None,
            efficiency: float | dict[float, float] = 1.0,
            abs_setpoint_deviation_kw: float | None = None,
            rel_setpoint_deviation: float | None = None,
            actuation_delay_s: float | None = None,
            wake_up_delay_s: float | None = None,
            set_delay_s: float | None = None,
            get_delay_s: float | None = None,
    ) -> None:
        self.name = name
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.e_required_kwh = e_required_kwh
        self.p_max_kw = p_max_kw
        self.battery_capacity_kwh = battery_capacity_kwh
        self.ramp_up_kw_pr_s = ramp_up_kw_pr_s
        self.ramp_down_kw_pr_s = ramp_down_kw_pr_s
        self.soc: float = init_soc
        self.actuation_delay: Delay | None = Delay.create(actuation_delay_s, name)
        self.wake_up_delay_s: float | None = self._init_wake_up_delay(
            actuation_delay_s, wake_up_delay_s
        )
        self.set_delay: Delay | None = Delay.create(set_delay_s, name)
        self.get_delay: Delay | None = Delay.create(get_delay_s, name)
        self.applied_p_setpoint: float = 0.0
        self.wake_up_due_t: float | None = None
        self.e_charged_kwh: float = 0.0
        self.p_cons_kw: float = 0.0
        self.next_p_cons_kw: float = 0.0
        self.efficiency = self._init_efficiency(efficiency)
        self.abs_setpoint_deviation: float | None = abs_setpoint_deviation_kw
        self.rel_setpoint_deviation: float | None = rel_setpoint_deviation
        self._init_set_server_data(server)

    def _init_wake_up_delay(
            self, actuation_delay_s: float | None, wake_up_delay_s: float | None
    ) -> float | None:
        if wake_up_delay_s is not None:
            if wake_up_delay_s < 0.0:
                msg = (
                    f"Wake-up delay for EV {self.name} is set to {wake_up_delay_s}. Wake-up delay "
                    f"cannot be negative."
                )
                raise ValueError(msg)
            return wake_up_delay_s
        elif actuation_delay_s is not None:
            return actuation_delay_s
        else:
            return None

    def _init_efficiency(
            self, efficiency: float | dict[float, float]
    ) -> float | dict[float, float]:
        if isinstance(efficiency, float):
            if efficiency <= 0.0 or efficiency > 1.0:
                msg = (
                    f"Efficiency for EV {self.name} is set to {efficiency}. Efficiency must be in "
                    f"the range (0, 1]."
                )
                raise ValueError(msg)
            return efficiency
        elif isinstance(efficiency, dict):
            # sort the dict by keys to ensure correct behavior when looking up efficiency based on
            # power consumption
            efficiency_dict_A = dict(sorted(efficiency.items()))
            # convert keys from A to kW
            efficiency_dict_kW = {
                convert_A_to_kw(i): eff for i, eff in efficiency_dict_A.items()
            }
            return efficiency_dict_kW
        else:
            msg = f"Invalid type for efficiency: {type(efficiency)}. Must be float or dict."
            raise TypeError(msg)

    def _init_set_server_data(self, server) -> None:
        server_data = {
            (self.name, "e_required_kwh"): self.e_required_kwh,
            (self.name, "arrival_time"): self.arrival_time,
            (self.name, "departure_time"): self.departure_time,
            (self.name, "next_p_cons_kw"): self.next_p_cons_kw,
            (self.name, "p_cons_kw"): self.p_cons_kw,
            (self.name, "e_charged_kwh"): self.e_charged_kwh,
            (self.name, "p_max_kw"): self.p_max_kw,
        }
        # No delay for the initial data, to ensure it's available to central and micro controllers
        # at t=0.
        server.set_from_keys(server_data=server_data)

    def get_efficiency(self) -> float:
        """Return charging efficiency at the current mean power level.

        Returns:
            Constant efficiency or the piecewise value for current mean power.
        """
        if isinstance(self.efficiency, float):
            return self.efficiency
        if isinstance(self.efficiency, dict):
            p_mean_cons_kw = (self.p_cons_kw + self.next_p_cons_kw) / 2
            # find the largest key that is <= p_mean_cons_kw
            keys = list(self.efficiency.keys())
            idx = bisect_right(keys, p_mean_cons_kw) - 1
            # if all keys are greater, use the smallest key's efficiency
            idx = max(idx, 0)
            return self.efficiency[keys[idx]]

    def update_soc(self, dt: float) -> None:
        """Integrate charged energy and state of charge over *dt* seconds.

        Args:
            dt: Simulation step duration in seconds.
        """
        # Update the state of charge (SoC)
        # Convert kW to kWh for the time step in seconds
        e_charged_t = (self.p_cons_kw + self.next_p_cons_kw) / 2 * (dt / 3600)
        self.e_charged_kwh += e_charged_t
        self.soc += e_charged_t * self.get_efficiency() / self.battery_capacity_kwh

    def is_not_connected(self, engine: Engine) -> bool:
        """Return whether this EV is unavailable for charging.

        Args:
            engine: Engine providing current time and charger assignments.

        Returns:
            ``True`` before arrival, after departure, once charged, or when no
            charger is assigned.
        """
        has_not_arrived = engine.state.t < self.arrival_time
        is_departed = engine.state.t >= self.departure_time
        is_charged = self.e_charged_kwh >= self.e_required_kwh
        has_no_charger = self.name not in engine.charger_by_ev_name
        return has_not_arrived or is_departed or is_charged or has_no_charger

    def apply_actuation_delay(self, t: float, p_setpoint_kw: float) -> None:
        """
        Schedule the new setpoint with actuation delay, and apply any expired setpoints. 
        Differentiate between wake-up delay and regular actuation delay with the following logic:
            - If the EV is currently not consuming power (next_p_cons_kw == 0) and receives a positive 
            setpoint, schedule the setpoint with the wake-up delay and set the wake-up due time.
            - If the EV is currently in wake-up delay, do not schedule new setpoints to avoid 
            overwriting the wake-up setpoint in the queue.
            - If the EV is awake or receives a zero setpoint, schedule with regular actuation delay.

        Args:
            t: Current simulated time in seconds.
            p_setpoint_kw: Charger-requested power in kilowatts.
        """
        if self.next_p_cons_kw == 0.0 and p_setpoint_kw > 0.0 and self.wake_up_due_t is None:
            self.actuation_delay.clear()
            self.actuation_delay.schedule(t, p_setpoint_kw, delay_s=self.wake_up_delay_s)
            self.wake_up_due_t = t + self.wake_up_delay_s
        elif self.wake_up_due_t is not None and t < self.wake_up_due_t:
            # wake-up pending
            pass
        else:
            self.actuation_delay.schedule(t, p_setpoint_kw)

        applied_value = self.actuation_delay.pop_expired(t)
        if applied_value is not None:
            self.applied_p_setpoint = applied_value
        if self.wake_up_due_t is not None and t >= self.wake_up_due_t:
            self.wake_up_due_t = None

    def calculate_power_consumption(self, dt: float) -> float:
        """Calculate next-step power after deviations, ramps, and limits.

        Args:
            dt: Simulation step duration in seconds.

        Returns:
            Bounded next-step power consumption in kilowatts.
        """
        p_setpoint_kw = self.applied_p_setpoint
        # Put setpoint to zero if it is below 6 A
        # if p_setpoint_kw < 4.157:  # p = sqrt(3) * V * I = sqrt(3) * 400 V * 6 A = 4.157 kW
        #     p_setpoint_kw = 0.0

        # Apply setpoint deviations if specified
        if self.rel_setpoint_deviation is not None:
            p_setpoint_kw += p_setpoint_kw * self.rel_setpoint_deviation
        if self.abs_setpoint_deviation is not None:
            p_setpoint_kw += self.abs_setpoint_deviation

        # Ramp up constraint
        if self.ramp_up_kw_pr_s is not None:
            max_ramp_up_kw = self.ramp_up_kw_pr_s * dt
            p_setpoint_kw = min(p_setpoint_kw, self.p_cons_kw + max_ramp_up_kw)

        # Ramp down constraint
        if self.ramp_down_kw_pr_s is not None:
            max_ramp_down_kw = self.ramp_down_kw_pr_s * dt
            p_setpoint_kw = max(p_setpoint_kw, self.p_cons_kw - max_ramp_down_kw)

        # Ensure power setpoint is within bounds
        p_setpoint_kw = min(p_setpoint_kw, self.p_max_kw)
        p_setpoint_kw = max(p_setpoint_kw, 0.0)

        return p_setpoint_kw

    def set_server_data(self, engine: Engine) -> None:
        """Publish EV state immediately or through its communication delay.

        Args:
            engine: Engine providing simulated time and the shared server.
        """
        server_data = {
            (self.name, "next_p_cons_kw"): self.next_p_cons_kw,
            (self.name, "p_cons_kw"): self.p_cons_kw,
            (self.name, "e_charged_kwh"): self.e_charged_kwh,
        }
        if self.set_delay is None:
            engine.server.set_from_keys(server_data=server_data)
        else:
            self.set_delay.schedule(engine.state.t, server_data)

    def step_set_delay(self, engine: Engine) -> None:
        """Publish EV state updates whose delay has expired.

        Args:
            engine: Engine providing simulated time and the shared server.
        """
        if self.set_delay is not None:
            delayed_data = self.set_delay.pop_expired(engine.state.t)
            if delayed_data is not None:
                engine.server.set_from_keys(server_data=delayed_data)

    def step(self, engine: Engine, dt: float) -> None:
        """
        Advance EV state by one time step using the setpoint from its charger.

        Args:
            engine: Engine providing charger, server, and timing state.
            dt: Simulation step duration in seconds.
        """
        self.update_soc(dt)

        if self.is_not_connected(engine):
            self.p_cons_kw = 0.0
            self.next_p_cons_kw = 0.0
        else:
            # read latest charger setpoint
            p_setpoint_kw = engine.charger_by_ev_name[self.name].p_setpoint

            if self.actuation_delay is None:
                self.applied_p_setpoint = p_setpoint_kw
            else:
                self.apply_actuation_delay(engine.state.t, p_setpoint_kw)

            # Step to the power consumption as calculated in previous step
            self.p_cons_kw = self.next_p_cons_kw

            # Calculate power consumption in next step based on the effective setpoint
            self.next_p_cons_kw = self.calculate_power_consumption(dt)

        # Publish data to the server
        self.set_server_data(engine)
        self.step_set_delay(engine)
