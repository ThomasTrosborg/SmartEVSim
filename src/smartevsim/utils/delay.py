"""Queue-based actuation delays and randomized update intervals."""

from __future__ import annotations

import heapq
import random


class Delay:
    """Delay scheduled values until their simulated activation time.

    Attributes:
        delay_s: Default delay in seconds.
        queue: Pending ``(activation_time, sequence, value)`` entries.
    """

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.queue: list[tuple[float, int, float]] = []
        self._seq = 0

    @classmethod
    def create(cls, delay_s: float | None, name: str) -> Delay | None:
        """Create a delay when enabled, validating that it is non-negative.

        Args:
            delay_s: Delay duration, or ``None``/zero to disable the delay.
            name: Component name included in validation errors.

        Returns:
            A delay instance when enabled, otherwise ``None``.

        Raises:
            ValueError: If ``delay_s`` is negative.
        """
        if delay_s is None or delay_s == 0.0:
            return None
        elif delay_s < 0.0:
            msg = f"Delay is attempted set to {delay_s} for unit {name}. Delay cannot be negative."
            raise ValueError(msg)
        else:
            return cls(delay_s)

    def schedule(self, t: float, value: float, delay_s: float | None = None) -> None:
        """Schedule a value relative to simulated time *t*.

        Args:
            t: Current simulated time in seconds.
            value: Value to deliver.
            delay_s: Optional one-off delay overriding :attr:`delay_s`.
        """
        act_t = t + (delay_s if delay_s is not None else self.delay_s)
        heapq.heappush(self.queue, (act_t, self._seq, value))
        self._seq += 1

    def pop_expired(self, t: float) -> float | None:
        """Return the latest value due by *t*, removing all due values.

        Args:
            t: Current simulated time in seconds.

        Returns:
            Most recently scheduled expired value, or ``None`` if none is due.
        """
        # find the latest pending setpoint whose (receive_time + delay) <= t
        applied_value = None
        while self.queue and self.queue[0][0] <= t:
            _, _, val = heapq.heappop(self.queue)
            applied_value = val  # keep overwriting so final value is the most recent expired

        return applied_value
    
    def clear(self) -> None:
        """Discard all scheduled values."""
        self.queue.clear()


class UpdateDelay:
    """Gate updates to a fixed interval with a randomized initial phase.

    Attributes:
        update_interval_s: Minimum time between updates, in seconds.
    """

    def __init__(self, update_interval_s: float, seed: int = 42) -> None:
        self.update_interval_s = update_interval_s
        self._last_update_t: float | None = None
        self._rng = random.Random(seed)

    @classmethod
    def create(
        cls, update_interval_s: float | None, name: str, seed: int = 42
    ) -> UpdateDelay | None:
        """Create an update gate when enabled, validating its interval.

        Args:
            update_interval_s: Update interval, or ``None``/zero to disable it.
            name: Component name included in validation errors.
            seed: Seed controlling the initial randomized phase.

        Returns:
            An update gate when enabled, otherwise ``None``.

        Raises:
            ValueError: If ``update_interval_s`` is negative.
        """
        if update_interval_s is None or update_interval_s == 0.0:
            return None
        elif update_interval_s < 0.0:
            msg = (
                f"Update interval is attempted set to {update_interval_s} for unit {name}. "
                f"Update interval cannot be negative."
            )
            raise ValueError(msg)
        else:
            return cls(update_interval_s, seed)

    def should_update(self, t: float) -> bool:
        """Return whether an update is due at simulated time *t*.

        Args:
            t: Current simulated time in seconds.

        Returns:
            ``True`` when the configured interval has elapsed.
        """
        # Initialize last update time with random offset to avoid synchronization when multiple
        # units have the same update interval.
        if self._last_update_t is None:
            self._last_update_t = t - self._rng.uniform(0, self.update_interval_s)
            return False
        elif (t - self._last_update_t) >= self.update_interval_s:
            self._last_update_t = t
            return True
        else:
            return False
