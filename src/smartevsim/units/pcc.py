"""Point-of-common-coupling power aggregation."""

from smartevsim.engine import Engine
from smartevsim.utils.delay import Delay, UpdateDelay


class PCC:
    """Measure aggregate EV load and publish it to the shared server.

    Attributes:
        name: Unique component name used for server keys.
        update_interval: Optional gate controlling measurement publication.
        set_delay: Optional communication delay for published measurements.
        p_total_kw: Most recently measured aggregate EV power.
    """

    def __init__(
            self,
            name: str,
            update_interval_s: float | None = None,
            set_delay_s: float | None = None,
            seed: int = 42,
    ) -> None:
        self.name = name
        self.update_interval = UpdateDelay.create(update_interval_s, name, seed=seed)
        self.set_delay: Delay | None = Delay.create(set_delay_s, name)
        self.p_total_kw: float = 0.0

    def set_server_data(self, engine: Engine) -> None:
        """Publish aggregate power immediately or schedule delayed delivery.

        Args:
            engine: Engine providing simulated time and the shared server.
        """
        server_data = {(self.name, "p_total_kw"): self.p_total_kw}
        if self.set_delay is None:
            engine.server.set_from_keys(server_data=server_data)
        else:
            self.set_delay.schedule(engine.state.t, server_data)

    def step_set_delay(self, engine: Engine) -> None:
        """Publish aggregate-power data whose communication delay has expired.

        Args:
            engine: Engine providing simulated time and the shared server.
        """
        if self.set_delay is not None:
            delayed_data = self.set_delay.pop_expired(engine.state.t)
            if delayed_data is not None:
                engine.server.set_from_keys(server_data=delayed_data)

    def step(self, engine: Engine) -> None:
        """Measure total EV power and advance the publication delay.

        Args:
            engine: Engine containing EV state, time, and the shared server.
        """
        # Update total power consumption at each time step but only publish to Server at specified 
        # update interval (or every step if update_interval is None)
        self.p_total_kw = sum(ev.p_cons_kw for ev in engine.evs.values())
        if self.update_interval is None or self.update_interval.should_update(engine.state.t):
            self.set_server_data(engine)

        self.step_set_delay(engine)
