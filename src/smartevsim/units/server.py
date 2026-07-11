"""In-memory communication server shared by simulation components."""

from collections import defaultdict

type ServerKey = tuple[str, str]
type ServerPriorityMap = dict[str, float]
type ServerValue = float | ServerPriorityMap
type ServerData = dict[ServerKey, ServerValue]
type ServerReadData = dict[ServerKey, ServerValue | None]


class Server:
    """Store and retrieve component variables by unit and variable name.

    Attributes:
        data: Nested mapping from unit names to variable names and values.
    """

    def __init__(self) -> None:
        self.data: defaultdict[str, defaultdict[str, ServerValue]] = defaultdict(
            lambda: defaultdict(dict)
        )

    def set(self, unit: str, variable: str, value: ServerValue) -> None:
        """Publish a value for a unit-variable pair.

        Args:
            unit: Publishing component name.
            variable: Published variable name.
            value: Scalar value or per-EV priority mapping.
        """
        self.data[unit][variable] = value

    def get(self, unit: str, variable: str) -> ServerValue | None:
        """Retrieve a published value.

        Args:
            unit: Publishing component name.
            variable: Published variable name.

        Returns:
            Stored value, or ``None`` when the key is unavailable.
        """
        # Return None if unit or variable is not found, or if value is not set.
        unit_data = self.data.get(unit)
        if unit_data is None:
            return None
        return unit_data.get(variable)

    def set_from_keys(self, server_data: ServerData) -> None:
        """Publish a mapping keyed by ``(unit, variable)`` tuples.

        Args:
            server_data: Values keyed by publishing unit and variable.
        """
        for key, value in server_data.items():
            self.set(*key, value)

    def get_from_keys(
            self, server_keys: list[ServerKey],
    ) -> ServerReadData:
        """Read several unit-variable keys into a single mapping.

        Args:
            server_keys: Unit-variable pairs to retrieve.

        Returns:
            Requested keys mapped to their stored values or ``None``.
        """
        server_get = self.get
        server_data = {key: server_get(*key) for key in server_keys}
        return server_data
