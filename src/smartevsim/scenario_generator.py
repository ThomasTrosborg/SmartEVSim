
"""Generate reproducible EV arrival and departure scenarios."""

import logging
import random

from smartevsim.utils.data_classes import SimConfig

logger = logging.getLogger(__name__)


class ScenarioGenerator:
    """Generate EV configurations for a fixed number of connection slots.

    Connection and idle durations are sampled from bounded normal
    distributions. Each vacated slot receives another EV until the simulation
    horizon is reached.

    Attributes:
        max_connected_evs: Number of reusable concurrent connection slots.
        avg_connection_time: Mean connection duration in seconds.
        avg_idle_time: Mean idle duration between EVs in a slot, in seconds.
        ev_configs: EV defaults-file names sampled for generated vehicles.
        e_required_kwh: Energy requested by each generated EV.
        init_soc: Initial state of charge assigned to each EV.
        sim_config: Simulation timing configuration.
        rng: Seeded random-number generator used for all sampling.
    """
    def __init__(
        self,
        max_connected_evs: int,
        avg_connection_time: float,
        avg_idle_time: float,
        ev_configs: list[str],
        e_required_kwh: float,
        init_soc: float,
        sim_config: SimConfig,
        seed: int = 42,
    ):
        self.max_connected_evs = max_connected_evs
        self.avg_connection_time = avg_connection_time
        self.avg_idle_time = avg_idle_time
        self.ev_configs = ev_configs
        self.e_required_kwh = e_required_kwh
        self.init_soc = init_soc
        self.sim_config = sim_config
        self.rng = random.Random(seed)

    def _sample_connection_time(self) -> float:
        std_dev = self.avg_connection_time / 4
        connection_time = self.rng.normalvariate(self.avg_connection_time, std_dev)
        return max(connection_time, 0.0)

    def _sample_idle_time(self) -> float:
        std_dev = self.avg_idle_time / 4
        idle_time = self.rng.normalvariate(self.avg_idle_time, std_dev)
        return max(idle_time, 0.0)

    def generate_ev_configs(self) -> list[dict]:
        """Generate per-EV configuration dictionaries for the simulation.

        Returns:
            EV constructor dictionaries ordered by creation time.

        Raises:
            ValueError: If slot counts, durations, or EV templates are invalid.
        """
        logger.info("Starting scenario generation.")
        if self.max_connected_evs < 1:
            msg = "max_connected_evs must be at least 1."
            raise ValueError(msg)
        if self.avg_connection_time <= 0.0:
            msg = "avg_connection_time must be positive."
            raise ValueError(msg)
        if self.avg_idle_time <= 0.0:
            msg = "avg_idle_time must be positive."
            raise ValueError(msg)
        if not self.ev_configs:
            msg = "ev_configs must contain at least one EV config."
            raise ValueError(msg)

        ev_configs = []
        slot_next_arrival_times = [
            self.rng.uniform(0.0, self.avg_connection_time)
            for _ in range(self.max_connected_evs)
        ]

        while True:
            slot_idx, arrival_time = min(
                enumerate(slot_next_arrival_times),
                key=lambda item: item[1],
            )
            if arrival_time >= self.sim_config.time_horizon:
                break

            connection_time = self._sample_connection_time()
            idle_time = self._sample_idle_time()
            departure_time = min(
                arrival_time + connection_time,
                self.sim_config.time_horizon,
            )
            ev_id = len(ev_configs) + 1
            ev_configs.append(
                {
                    "name": f"ev_{ev_id}",
                    "arrival_time": arrival_time,
                    "departure_time": departure_time,
                    "e_required_kwh": self.e_required_kwh,
                    "init_soc": self.init_soc,
                    "ev_config": self.rng.choice(self.ev_configs),
                }
            )

            slot_next_arrival_times[slot_idx] = departure_time + idle_time

        logger.info("Scenario generation completed.")
        return ev_configs
