"""Priority-allocation strategies for EV charging controllers."""

from smartevsim.engine import Engine


def get_connected_ev_names(engine: Engine, t: float) -> list[str]:
    """Return names of EVs connected at simulated time *t*.

    Args:
        engine: Engine containing the EV population.
        t: Simulated time in seconds.

    Returns:
        Names whose EV arrival and departure interval contains ``t``.
    """
    connected_ev_names = [
        ev_name
        for ev_name, ev in engine.evs.items()
        if ev.arrival_time <= t < ev.departure_time
    ]
    return connected_ev_names

def calculate_total_urgency_priority(engine: Engine, t: float) -> float:
    """
    Calculate and sum the absolute priority of each EV based on the
    remaining energy to be charged and time to departure.

    abs_priority = (e_required_kwh - e_charged_kwh) / (t_departure - t)

    Args:
        engine: Engine containing EV energy and timing state.
        t: Simulated time in seconds.

    Returns:
        Sum of absolute urgency values for connected EVs.
    """
    connected_ev_names = get_connected_ev_names(engine, t)
    total_priority = sum(
        (
            engine.evs[ev_name].e_required_kwh - engine.evs[ev_name].e_charged_kwh
        ) / (engine.evs[ev_name].departure_time - t)
        for ev_name in connected_ev_names
    )
    return total_priority

def calculate_urgency_priority(
        ev_name: str, engine: Engine, total_priority: float | None = None
) -> float:
    """Return one EV's normalized remaining-energy urgency.

    Args:
        ev_name: Name of the EV to evaluate.
        engine: Engine containing EV energy and timing state.
        total_priority: Optional precomputed total absolute urgency.

    Returns:
        EV urgency divided by total urgency, or zero for a zero total.
    """
    if total_priority is None:
        total_priority = calculate_total_urgency_priority(engine, engine.state.t)

    ev = engine.evs[ev_name]
    abs_priority_i = (
        (ev.e_required_kwh - ev.e_charged_kwh) / (ev.departure_time - engine.state.t)
    )
    rel_priority_i = abs_priority_i / total_priority if total_priority > 0 else 0.0
    return rel_priority_i


def calculate_urgency_priorities(engine: Engine, t: float) -> dict[str, float]:
    """
    Calculate the relative urgency priority of each EV based on the remaining energy to be charged.
    abs_priority = (e_required_kwh - e_charged_kwh) / (t_departure - t)
    rel_priority = abs_priority / sum(abs_priorities) for all EVs

    Args:
        engine: Engine containing EV energy and timing state.
        t: Simulated time in seconds.

    Returns:
        Mapping from connected EV names to normalized urgency weights.
    """
    connected_ev_names = get_connected_ev_names(engine, t)
    if not connected_ev_names:
        return {}

    ev_names: list[str] = []
    abs_values: list[float] = []

    for ev_name in connected_ev_names:
        ev = engine.evs[ev_name]
        abs_prio = (ev.e_required_kwh - ev.e_charged_kwh) / (ev.departure_time - t)
        ev_names.append(ev_name)
        abs_values.append(abs_prio)

    total_abs_priority = sum(abs_values)
    if total_abs_priority <= 0.0:
        return dict.fromkeys(ev_names, 0.0)

    priorities = {
        ev_name: abs_prio / total_abs_priority
        for ev_name, abs_prio in zip(ev_names, abs_values, strict=True)
    }
    return priorities

def get_ev_names(engine: Engine) -> list[str]:
    """Select EV names according to the engine's priority configuration.

    Args:
        engine: Engine whose priority configuration selects the population.

    Returns:
        Names of all EVs or the currently connected EVs.

    Raises:
        ValueError: If the population selector is invalid.
    """
    if engine.priority_config.evs == "all":
        return list(engine.evs.keys())
    elif engine.priority_config.evs == "connected" or isinstance(engine.priority_config.evs, int):
        return get_connected_ev_names(engine, engine.state.t)
    else:
        msg = (
            f"Invalid priority_config.evs value: {engine.priority_config.evs}. Expected 'all', "
            f"'connected', or a non-negative integer."
        )
        raise ValueError(msg)

def calculate_even_priority(engine: Engine) -> float:
    """Return the equal priority weight assigned to one selected EV.

    Args:
        engine: Engine whose priority configuration selects the population.

    Returns:
        Reciprocal of the selected or configured EV count, or zero if empty.
    """
    if isinstance(engine.priority_config.evs, int):
        n = engine.priority_config.evs
    else:
        n = len(get_ev_names(engine))
    if n > 0:
        return 1.0 / n
    else:
        return 0.0

def calculate_even_priorities(engine: Engine) -> dict[str, float]:
    """Return equal priority weights for all selected EVs.

    Args:
        engine: Engine whose priority configuration selects the population.

    Returns:
        Mapping from selected EV names to equal weights.
    """
    ev_names = get_ev_names(engine)
    if isinstance(engine.priority_config.evs, int):
        n = engine.priority_config.evs
    else:
        n = len(ev_names)
    if n > 0:
        return dict.fromkeys(ev_names, 1.0 / n)
    else:
        return {}
