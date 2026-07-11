

"""EV charger models and setpoint communication variants."""

import math
from abc import ABC, abstractmethod

from smartevsim.engine import Engine
from smartevsim.utils.conversions import convert_A_to_kw, convert_kw_to_A
from smartevsim.utils.delay import Delay


class Charger(ABC):
    """Base charger that validates, rounds, and delays power setpoints.

    A charger reads a controller setpoint, converts it through configurable
    current constraints, and exposes the resulting EV-facing power setpoint.
    Communication and EVSE actuation delays are modeled independently.

    Attributes:
        name: Unique charger name.
        ev_name: Name of the associated EV.
        mc_name: Name of the associated micro-controller.
        p_setpoint: Current EV-facing power setpoint in kilowatts.
        i_min: Minimum nonzero charging current in amperes.
        i_rounding_method: Current rounding mode: ``round``, ``floor``, or
            ``ceil``.
        i_granularity: Allowed current increment in amperes.
        set_delay: Optional publication delay.
        get_delay: Optional server read delay.
        evse_delay: Optional physical EVSE actuation delay.
        server_data: Most recently received input values.
    """

    def __init__(
            self,
            name: str,
            ev_name: str,
            mc_name: str,
            init_p_setpoint: float = 0.0,
            set_delay_s: float | None = None,
            get_delay_s: float | None = None,
            evse_delay_s: float | None = None,
            i_min: float = 6.0,
            i_rounding_method: str = "floor",
            i_granularity: float = 1.0,
    ) -> None:
        self.name = name
        self.ev_name = ev_name
        self.mc_name = mc_name
        self.p_setpoint = init_p_setpoint
        self.i_min = i_min
        self.i_rounding_method = i_rounding_method
        self.i_granularity = i_granularity
        self.rel_priority_i: float | None = None
        self.set_delay: Delay | None = Delay.create(set_delay_s, name)
        self.get_delay: Delay | None = Delay.create(get_delay_s, name)
        self.evse_delay: Delay | None = Delay.create(evse_delay_s, name)
        self.server_data: dict[tuple[str, str], float] | None = None
        self.server_get_keys: list = [(ev_name, "arrival_time"), (ev_name, "departure_time")]

    def get_server_data(self, engine: Engine) -> None:
        """Read required values from the server, applying the read delay.

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

    def has_required_inputs(self) -> bool:
        """Return whether all required server values are available.

        Returns:
            ``True`` when server data exists and contains no missing values.
        """
        return self.server_data is not None and None not in self.server_data.values()

    def round_to_integer_current_setpoint(self, p_setpoint_kw: float) -> float:
        """
        Round the current setpoint according to the specified rounding method and granularity, and
        apply the minimum current constraint.

        Args:
            p_setpoint_kw: Unconstrained power setpoint in kilowatts.

        Returns:
            Power setpoint corresponding to the constrained current.

        Raises:
            ValueError: If the configured rounding method is unsupported.
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
                f"Invalid current rounding method in microcontroller {self.name}: "
                f"{self.i_rounding_method}. Expected 'round', 'floor' or 'ceil'."
            )
            raise ValueError(msg)
        if i_setpoint < self.i_min:
            i_setpoint = 0.0
        p_setpoint_kw = convert_A_to_kw(i_setpoint)
        return p_setpoint_kw

    def apply_evse_delay(self, t: float, p_setpoint: float) -> None:
        """Apply a setpoint immediately or queue it for EVSE actuation.

        Args:
            t: Current simulated time in seconds.
            p_setpoint: Constrained power setpoint in kilowatts.
        """
        if self.evse_delay is None:
            self.p_setpoint = p_setpoint
        else:
            self.evse_delay.schedule(t, p_setpoint)

    def step_evse_delay(self, t: float) -> None:
        """Apply the latest EVSE setpoint whose delay has expired.

        Args:
            t: Current simulated time in seconds.
        """
        if self.evse_delay is not None:
            delayed_setpoint = self.evse_delay.pop_expired(t)
            if delayed_setpoint is not None:
                self.p_setpoint = delayed_setpoint

    @abstractmethod
    def read_setpoint(self, engine: Engine) -> float:
        """Read the raw controller setpoint for this charger.

        Args:
            engine: Engine providing controller and server state.

        Returns:
            Requested EV power in kilowatts.
        """
        msg = "Subclasses of Charger must implement abstract method read_setpoint."
        raise NotImplementedError(msg)

    def step(self, engine: Engine) -> None:
        """Update the charger setpoint for one simulation step.

        Args:
            engine: Engine providing time, controller state, and server data.
        """
        # Always advance read-delay state each step so delayed data reception is
        # decoupled from the controller update interval.
        self.get_server_data(engine)

        if self.has_required_inputs() and self.is_connected(engine.state.t):
            p_setpoint = self.read_setpoint(engine)
            p_setpoint = self.round_to_integer_current_setpoint(p_setpoint)
            self.apply_evse_delay(engine.state.t, p_setpoint)
        else:
            self.p_setpoint = 0.0
            return

        self.step_evse_delay(engine.state.t)


class IntegratedCharger(Charger):
    """
    The IntegratedCharger has the microcontroller integrated into the charger which means the 
    communication delay between the microcontroller and the charger is negligible. Hence, this 
    charger reads the setpoint directly from the microcontroller. Note that the charger may still 
    have an EVSE delay. 

    Attributes are inherited from :class:`Charger`.
    """

    def read_setpoint(self, engine: Engine) -> float:
        """Read the setpoint directly from the integrated micro-controller.

        Args:
            engine: Engine containing the micro-controller lookup.

        Returns:
            Associated micro-controller setpoint in kilowatts.
        """
        return engine.mc_by_ev_name[self.ev_name].p_setpoint


class ExternalCharger(Charger):
    """
    The ExternalCharger is a charger where the microcontroller is external, which means there may 
    be a communication delay between the microcontroller and the charger. Hence, this charger reads
    the setpoint from the server data which is updated by the microcontroller.

    Attributes are inherited from :class:`Charger`.
    """

    def __init__(
            self,
            name: str,
            ev_name: str,
            mc_name: str,
            init_p_setpoint: float = 0.0,
            set_delay_s: float | None = None,
            get_delay_s: float | None = None,
            evse_delay_s: float | None = None,
            i_min: float = 6.0,
            i_rounding_method: str = "floor",
            i_granularity: float = 1.0,
    ) -> None:
        super().__init__(
            name=name,
            ev_name=ev_name,
            mc_name=mc_name,
            init_p_setpoint=init_p_setpoint,
            set_delay_s=set_delay_s,
            get_delay_s=get_delay_s,
            evse_delay_s=evse_delay_s,
            i_min=i_min,
            i_rounding_method=i_rounding_method,
            i_granularity=i_granularity,
        )
        self.server_get_keys.append((self.mc_name, "p_setpoint"))

    def read_setpoint(self, engine: Engine) -> float:
        """Read the associated micro-controller setpoint from server data.

        Args:
            engine: Engine context; unused because delayed data is held locally.

        Returns:
            Received micro-controller setpoint in kilowatts.

        Raises:
            ValueError: If server data or the expected setpoint is unavailable.
        """
        if self.server_data is None:
            msg = f"ExternalCharger {self.name} has no server data available to read setpoint from."
            raise ValueError(msg)
        if (self.mc_name, "p_setpoint") not in self.server_data:
            msg = (
                f"ExternalCharger {self.name} has no server data available for microcontroller "
                f"{self.mc_name}."
            )
            raise ValueError(msg)
        return self.server_data[(self.mc_name, "p_setpoint")]
