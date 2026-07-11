"""Compute tracking, overload, response-time, and consensus metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from smartevsim.utils.conversions import convert_A_to_kw


@dataclass(frozen=True)
class TrackingErrorStats:
    """Summary statistics for aggregate setpoint tracking error.

    Attributes:
        mean_abs_kw: Mean absolute tracking error in kilowatts.
        rmse_kw: Root mean square tracking error in kilowatts.
        max_abs_kw: Maximum absolute tracking error in kilowatts.
    """
    mean_abs_kw: float
    rmse_kw: float
    max_abs_kw: float


@dataclass(frozen=True)
class OverloadStats:
    """Frequency, duration, and magnitude statistics for overload events.

    Attributes:
        frequency: Fraction of samples classified as overloaded.
        event_count: Number of distinct overload periods.
        total_duration_s: Combined overload duration in seconds.
        max_duration_s: Longest overload duration in seconds.
        mean_magnitude_kw: Mean positive overload magnitude in kilowatts.
        max_magnitude_kw: Maximum overload magnitude in kilowatts.
    """
    frequency: float
    event_count: int
    total_duration_s: float
    max_duration_s: float
    mean_magnitude_kw: float
    max_magnitude_kw: float


@dataclass(frozen=True)
class Metrics:
    """Top-level performance metrics for one simulation run.

    Attributes:
        mean_consensus_gap: Mean range of eligible priority ratios.
        mean_consensus_mad: Mean absolute deviation of eligible ratios.
        total_energy_kwh: Aggregate delivered energy.
        mean_abs_tracking_error_kw: Mean absolute aggregate tracking error.
        max_tracking_error_kw: Maximum absolute aggregate tracking error.
        overload: Overload event statistics.
        response_time_arrival: Tracking response times after arrivals.
        response_time_departure: Tracking response times after departures.
        response_time_to_consensus_arrival: Consensus times after arrivals.
        response_time_to_consensus_departure: Consensus times after departures.
        consensus_fail_frequency: Fraction of events that fail to converge.
    """
    mean_consensus_gap: float
    mean_consensus_mad: float
    total_energy_kwh: float
    mean_abs_tracking_error_kw: float
    max_tracking_error_kw: float
    overload: OverloadStats
    response_time_arrival: list[float]
    response_time_departure: list[float]
    response_time_to_consensus_arrival: list[float]
    response_time_to_consensus_departure: list[float]
    consensus_fail_frequency: float


def load_records(path: str | Path) -> pd.DataFrame:
    """Load simulation records from a CSV file.

    Args:
        path: CSV file containing wide simulation records.

    Returns:
        Loaded records with columns preserved from the CSV.
    """
    path = Path(path)
    return pd.read_csv(path)


def infer_dt_s(df: pd.DataFrame) -> float:
    """Infer the sampling interval from the median time difference.

    Args:
        df: Records containing a numeric ``t`` column.

    Returns:
        Median difference between consecutive valid time values, in seconds.

    Raises:
        ValueError: If the DataFrame has no ``t`` column.
    """
    if "t" not in df.columns:
        msg = "Cannot infer dt_s without a 't' column."
        raise ValueError(msg)
    t_values = pd.to_numeric(df["t"], errors="coerce").dropna()
    steps = t_values.diff().dropna()
    return float(steps.median())


def compute_metrics(
    df: pd.DataFrame,
    *,
    dt_s: float | None = None,
    epsilon_kw: float = 1.0,
    consensus_window_s: float = 3.0,
    consensus_gap_threshold: float = 0.05,
) -> tuple[dict[str, pd.DataFrame | pd.Series], Metrics]:
    """Compute summary metrics and intermediate consensus time series.

    Args:
        df: Wide simulation-record DataFrame.
        dt_s: Sampling interval in seconds, inferred when omitted.
        epsilon_kw: Tracking tolerance used for response times.
        consensus_window_s: Duration for which convergence must persist.
        consensus_gap_threshold: Maximum gap considered consensus.

    Returns:
        A pair containing consensus time series and summary metrics.
    """
    _validate_columns(df, {"p_total_kw"})
    _validate_columns(df, {"p_central_setpoint_kw"})
    if dt_s is None:
        dt_s = infer_dt_s(df)

    # Consensus metrics
    cons_priority_ratios = _calculate_cons_priority_ratios(df)
    consensus_gap, consensus_mad, consensus_participant_count = _compute_consensus_metrics(
        cons_priority_ratios, df
    )

    total_energy_kwh = float((df["p_total_kw"] * (dt_s / 3600)).sum())

    tracking_error = df["p_total_kw"] - df["p_central_setpoint_kw"]
    tracking_error_stats = TrackingErrorStats(
        mean_abs_kw=float(tracking_error.abs().mean()),
        rmse_kw=float(np.sqrt((tracking_error**2).mean())),
        max_abs_kw=float(tracking_error.abs().max()),
    )

    overload_amount = df["p_total_kw"] - df["p_central_setpoint_kw"]
    overload_mask = overload_amount > convert_A_to_kw(1.0) # 0
    overload_stats = _compute_overload_stats(overload_amount, overload_mask, dt_s)

    response_time_arrival = _compute_response_times(
        df,
        event_type="arrival",
        epsilon_kw=epsilon_kw,
        consensus_window_s=consensus_window_s,
        dt_s=dt_s,
    )
    response_time_departure = _compute_response_times(
        df,
        event_type="departure",
        epsilon_kw=epsilon_kw,
        consensus_window_s=consensus_window_s,
        dt_s=dt_s,
    )
    (
        response_time_to_consensus_arrival,
        arrival_consensus_fail_count,
        arrival_event_count,
    ) = _compute_response_times_to_consensus_gap(
        df,
        consensus_gap,
        consensus_participant_count,
        event_type="arrival",
        consensus_gap_threshold=consensus_gap_threshold,
        consensus_window_s=consensus_window_s,
        dt_s=dt_s,
    )
    (
        response_time_to_consensus_departure,
        departure_consensus_fail_count,
        departure_event_count,
    ) = _compute_response_times_to_consensus_gap(
        df,
        consensus_gap,
        consensus_participant_count,
        event_type="departure",
        consensus_gap_threshold=consensus_gap_threshold,
        consensus_window_s=consensus_window_s,
        dt_s=dt_s,
    )
    consensus_fail_count = arrival_consensus_fail_count + departure_consensus_fail_count
    event_count = arrival_event_count + departure_event_count
    consensus_fail_frequency = (
        consensus_fail_count / event_count if event_count > 0 else 0.0
    )

    consensus_dfs = {
        "cons_priority_ratios": cons_priority_ratios,
        "consensus_gap": consensus_gap,
        "consensus_mad": consensus_mad,
        "consensus_participant_count": consensus_participant_count,
    }
    metrics = Metrics(
        mean_consensus_gap=consensus_gap.mean(),
        mean_consensus_mad=consensus_mad.mean(),
        total_energy_kwh=total_energy_kwh,
        mean_abs_tracking_error_kw=tracking_error_stats.mean_abs_kw,
        max_tracking_error_kw=tracking_error_stats.max_abs_kw,
        overload=overload_stats,
        response_time_arrival=response_time_arrival,
        response_time_departure=response_time_departure,
        response_time_to_consensus_arrival=response_time_to_consensus_arrival,
        response_time_to_consensus_departure=response_time_to_consensus_departure,
        consensus_fail_frequency=consensus_fail_frequency,
    )

    return consensus_dfs, metrics


def _calculate_cons_priority_ratios(df: pd.DataFrame) -> pd.DataFrame:
    rel_priority_cols = _select_columns_with_prefix(df, "rel_priority_")
    p_ref = df["p_central_setpoint_kw"]
    # infer EV names from column names by removing prefix
    ev_names = [col[len("rel_priority_ev_") :] for col in rel_priority_cols]
    ratios = pd.DataFrame(index=df.index)
    for ev_name in ev_names:
        p_cons = df[f"p_cons_ev_{ev_name}"]
        rel_priority = df[f"rel_priority_ev_{ev_name}"]
        ratio_series = p_cons / (rel_priority * p_ref)
        ratios[ev_name] = ratio_series
    return ratios


def _compute_consensus_metrics(
    cons_priority_ratios: pd.DataFrame,
    df: pd.DataFrame,
    *,
    min_consumption_kw: float = 0, # 4.157,
    max_consumption_kw: float = 11.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    p_cons_cols = [f"p_cons_ev_{ev_name}" for ev_name in cons_priority_ratios.columns]
    p_cons_values = df[p_cons_cols].apply(pd.to_numeric, errors="coerce")
    p_cons_values.columns = cons_priority_ratios.columns

    eligible = (p_cons_values > min_consumption_kw) # & (p_cons_values < max_consumption_kw)
    eligible_ratios = cons_priority_ratios.where(eligible)

    participant_count = eligible_ratios.notna().sum(axis=1)

    # compute the consensus gap as the difference between the max and min ratio among eligible EVs
    # at each time step
    max_ratio = eligible_ratios.max(axis=1)
    min_ratio = eligible_ratios.min(axis=1)
    gap = max_ratio - min_ratio

    # copmute the consensus MAD as the mean absolute deviation of the ratios from their mean among 
    # eligible EVs at each time step
    mean_ratio = eligible_ratios.mean(axis=1)
    mad = (eligible_ratios.sub(mean_ratio, axis=0).abs().mean(axis=1))

    return gap, mad, participant_count


def _compute_overload_stats(
    overload_amount: pd.Series, overload_mask: pd.Series, dt_s: float
) -> OverloadStats:
    if overload_mask.sum() == 0:
        return OverloadStats(
            frequency=0.0,
            event_count=0,
            total_duration_s=0.0,
            max_duration_s=0.0,
            mean_magnitude_kw=0.0,
            max_magnitude_kw=0.0,
        )

    frequency = float(overload_mask.mean())
    event_count = int((overload_mask & ~overload_mask.shift(1, fill_value=False)).sum())

    total_duration_s = float(overload_mask.sum() * dt_s)
    max_duration_s = float(_max_run_length(overload_mask) * dt_s)

    overload_values = overload_amount[overload_mask]
    mean_magnitude_kw = float(overload_values.mean())
    max_magnitude_kw = float(overload_values.max())

    return OverloadStats(
        frequency=frequency,
        event_count=event_count,
        total_duration_s=total_duration_s,
        max_duration_s=max_duration_s,
        mean_magnitude_kw=mean_magnitude_kw,
        max_magnitude_kw=max_magnitude_kw,
    )


def _compute_response_times(
    df: pd.DataFrame,
    *,
    event_type: str,
    epsilon_kw: float,
    consensus_window_s: float,
    dt_s: float,
) -> list[float]:
    if "t" not in df.columns:
        msg = "Cannot compute response times without a 't' column."
        raise ValueError(msg)
    if "n_active" not in df.columns:
        msg = "Cannot compute response times without a 'n_active' column."
        raise ValueError(msg)

    p_cons_cols = _select_columns_with_prefix(df, "p_cons_ev_")
    if not p_cons_cols:
        return []

    t_series = pd.to_numeric(df["t"], errors="coerce").to_numpy()
    cons_values = df[p_cons_cols].apply(pd.to_numeric, errors="coerce")
    window_steps = _resolve_window_steps(consensus_window_s, dt_s)
    consensus_window_mask = _window_all_true_by_ev(cons_values, window_steps, epsilon_kw)

    event_indices = _collect_event_indices(df, event_type)
    response_times: list[float] = []
    consensus_array = consensus_window_mask
    for idx in event_indices:
        start_idx = idx + window_steps - 1
        if start_idx >= consensus_array.size:
            response_times.append(float("nan"))
            continue

        consensus_after = np.flatnonzero(consensus_array[start_idx:])
        if consensus_after.size == 0:
            response_times.append(float("nan"))
            continue

        consensus_idx = start_idx + int(consensus_after[0])
        window_start_idx = consensus_idx - (window_steps - 1)
        response_times.append(float(t_series[window_start_idx] - t_series[idx]))

    return response_times


def _compute_response_times_to_consensus_gap(
    df: pd.DataFrame,
    consensus_gap: pd.Series,
    consensus_participant_count: pd.Series,
    *,
    event_type: str,
    consensus_gap_threshold: float,
    consensus_window_s: float,
    dt_s: float,
) -> tuple[list[float], int, int]:
    if "t" not in df.columns:
        msg = "Cannot compute response times without a 't' column."
        raise ValueError(msg)
    if "n_active" not in df.columns:
        msg = "Cannot compute response times without a 'n_active' column."
        raise ValueError(msg)

    t_series = pd.to_numeric(df["t"], errors="coerce").to_numpy()
    n_active = pd.to_numeric(df["n_active"], errors="coerce").fillna(0.0)
    window_steps = _resolve_window_steps(consensus_window_s, dt_s)
    enough_participants = consensus_participant_count >= n_active.clip(lower=2)
    in_consensus = (consensus_gap <= consensus_gap_threshold) & enough_participants
    consensus_window_mask = _window_all_true(in_consensus, window_steps)

    event_indices = _collect_event_indices(df, event_type)
    all_event_indices = _collect_all_event_indices(df)
    event_count = len(event_indices)
    response_times: list[float] = []
    consensus_fail_count = 0
    for idx in event_indices:
        next_event_idx = _next_event_index(idx, all_event_indices, consensus_window_mask.size)
        interval_n_active = n_active.iloc[idx:next_event_idx]
        if interval_n_active.empty or interval_n_active.max() < 2:
            continue

        start_idx = idx + window_steps - 1
        if start_idx >= next_event_idx:
            consensus_fail_count += 1
            continue

        consensus_after = np.flatnonzero(consensus_window_mask[start_idx:next_event_idx])
        if consensus_after.size == 0:
            consensus_fail_count += 1
            continue

        consensus_idx = start_idx + int(consensus_after[0])
        window_start_idx = consensus_idx - (window_steps - 1)
        response_times.append(float(t_series[window_start_idx] - t_series[idx]))

    return response_times, consensus_fail_count, event_count


def _collect_all_event_indices(df: pd.DataFrame) -> list[int]:
    values = pd.to_numeric(df["n_active"], errors="coerce").fillna(0.0)
    prev = values.shift(1, fill_value=0.0)
    mask = values != prev
    return [int(idx) for idx in np.flatnonzero(mask.to_numpy())]


def _next_event_index(idx: int, event_indices: list[int], fallback_idx: int) -> int:
    later_event_indices = [event_idx for event_idx in event_indices if event_idx > idx]
    if not later_event_indices:
        return fallback_idx
    return later_event_indices[0]


def _collect_event_indices(df: pd.DataFrame, event_type: str) -> list[int]:
    if event_type not in {"arrival", "departure"}:
        msg = f"Unsupported event_type: {event_type}"
        raise ValueError(msg)

    values = pd.to_numeric(df["n_active"], errors="coerce").fillna(0.0)
    prev = values.shift(1, fill_value=0.0)
    mask = values > prev if event_type == "arrival" else values < prev
    return [int(idx) for idx in np.flatnonzero(mask.to_numpy())]


def _select_columns_with_prefix(df: pd.DataFrame, prefix: str) -> list[str]:
    return [col for col in df.columns if col.startswith(prefix)]


def _resolve_window_steps(consensus_window_s: float, dt_s: float) -> int:
    window_steps = int(np.ceil(consensus_window_s / dt_s))
    return max(1, window_steps)


def _window_all_true(mask: pd.Series, window_steps: int) -> np.ndarray:
    if window_steps <= 1:
        return mask.to_numpy()
    series = mask.astype(float)
    return (series.rolling(window_steps, min_periods=window_steps).mean() == 1.0).to_numpy()


def _window_all_true_by_ev(
    cons_values: pd.DataFrame, window_steps: int, epsilon_kw: float
) -> np.ndarray:
    if window_steps <= 1:
        window_min = cons_values
        window_max = cons_values
    else:
        window_min = cons_values.rolling(window_steps, min_periods=window_steps).min()
        window_max = cons_values.rolling(window_steps, min_periods=window_steps).max()
    per_ev_mask = (window_max - window_min) <= epsilon_kw
    return per_ev_mask.all(axis=1).to_numpy()


def _max_run_length(mask: pd.Series) -> int:
    if mask.sum() == 0:
        return 0
    group_ids = mask.ne(mask.shift(1, fill_value=False)).cumsum()
    run_lengths = mask.groupby(group_ids).sum()
    return int(run_lengths.max())


def _validate_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(df.columns)
    if missing:
        msg = f"Missing required columns: {sorted(missing)}"
        raise ValueError(msg)


def _format_significant_digits(value: float, significant_digits: int) -> str:
    if significant_digits <= 0:
        msg = "significant_digits must be positive."
        raise ValueError(msg)
    if not np.isfinite(value):
        return str(value)
    if value == 0:
        return "0"

    abs_value = abs(value)
    decimals = max(significant_digits - int(np.floor(np.log10(abs_value))) - 1, 0)
    formatted = f"{value:.{decimals}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def print_metrics(metrics: Metrics) -> None:
    """Print a human-readable summary and JSON representation of metrics.

    Args:
        metrics: Summary metrics to format and print.
    """
    sig3 = lambda value: _format_significant_digits(value, 3)
    all_response_times = metrics.response_time_to_consensus_arrival + metrics.response_time_to_consensus_departure
    finite_response_times = [value for value in all_response_times if np.isfinite(value)]
    mean_time_to_consensus = [float(np.mean(finite_response_times))] if finite_response_times else []
    string = (
        f"& {_format_list([metrics.mean_abs_tracking_error_kw], sig3)} "
        f"& {sig3(metrics.total_energy_kwh)} "
        f"& {metrics.overload.frequency * 100:.2f} "
        f"& {metrics.overload.total_duration_s:.0f} "
        f"& {metrics.overload.max_duration_s:.0f} "
        f"& {metrics.overload.mean_magnitude_kw:.2f} "
        f"& {_format_list([metrics.overload.max_magnitude_kw], sig3)} "
        f"& {metrics.mean_consensus_gap:.3f} "
        f"& {metrics.mean_consensus_mad:.3f} "
        f"& {_format_list(mean_time_to_consensus, sig3)} "
        f"& {(1 - metrics.consensus_fail_frequency) * 100:.2f} "
    )
    print(string)


def _format_list(values: list[float], formatter) -> str:
    if not values:
        return "-"
    return ", ".join(formatter(value) for value in values)

if __name__ == "__main__":

    """"case study silent consensus configs"""
    # results_path = "results/case_study/test_0_SC.csv"
    # results_path = "results/case_study/test_1_SC.csv"
    # results_path = "results/case_study/test_2_SC.csv"
    # results_path = "results/case_study/test_3_SC.csv"
    # results_path = "results/case_study/test_4_SC.csv"
    # results_path = "results/case_study/test_4_SC copy.csv"
    # results_path = "results/case_study/test_4_SC copy 2.csv"
    # results_path = "results/case_study/test_5_SC_30.csv"
    # results_path = "results/case_study/test_5_SC_100.csv"
    # results_path = "results/case_study/test_5_SC_1000.csv"

    """"case study centralized configs"""
    results_path = "results/case_study/test_0_Central.csv"
    # results_path = "results/case_study/test_1_Central.csv"
    # results_path = "results/case_study/test_2_Central.csv"
    # results_path = "results/case_study/test_3_Central.csv"
    # results_path = "results/case_study/test_4_Central.csv"
    # results_path = "results/case_study/test_4_Central copy.csv"
    # results_path = "results/case_study/test_4_Central copy 2.csv"
    # results_path = "results/case_study/test_5_Central.csv"

    # results_path = "results/test_default_1.csv"
    df = load_records(results_path)
    consensus_dfs, metrics = compute_metrics(
        df,
        dt_s=0.01,
        epsilon_kw=1.0,
        consensus_window_s=5.0,
        consensus_gap_threshold=0.15,
    )
    print(json.dumps(asdict(metrics), indent=2))
    print_metrics(metrics)
    print(0)
