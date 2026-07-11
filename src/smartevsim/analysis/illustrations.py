from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from smartevsim.analysis.metrics import compute_metrics, infer_dt_s, load_records


EV_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "ev_configs"
EV_EFFICIENCY_CONFIGS = {
    "tesla": EV_CONFIG_DIR / "tesla_model3_LRDM_2020.yaml",
    "vw": EV_CONFIG_DIR / "vw_id3_pro_2022.yaml",
    "skoda": EV_CONFIG_DIR / "skoda_enyaq_iV60_2021.yaml",
}
EV_COLORS = {"tesla": "#288e28", "vw": "#f68300", "skoda": "#4a4cd2"}


@dataclass(frozen=True)
class RunSpec:
    path: str | Path
    label: str | None = None
    color: str | None = None


@dataclass(frozen=True)
class RunData:
    path: Path
    label: str
    color: str
    df: pd.DataFrame


def load_runs(run_specs: Sequence[RunSpec]) -> list[RunData]:
    if not run_specs:
        return []

    colors = list(qualitative.Plotly)
    loaded: list[RunData] = []
    for idx, spec in enumerate(run_specs):
        path = Path(spec.path)
        label = spec.label or path.stem
        color = spec.color or colors[idx % len(colors)]
        df = load_records(path)
        loaded.append(RunData(path=path, label=label, color=color, df=df))
    return loaded


def plot_overload_histogram(
    runs: Sequence[RunData],
    *,
    bins: int | str = "auto",
    show: bool = True,
    save: bool = False,
    name: str | None = None,
    output_dir: str | Path = "figures",
) -> go.Figure:
    fig = go.Figure()
    for run in runs:
        overload = _overload_magnitude_kw(run.df)
        values = overload[overload > 0]
        bar = _build_histogram_bar(values, run.df, bins=bins)
        fig.add_trace(
            go.Scatter(
                x=bar["centers"],
                y=bar["durations"],
                mode="lines",
                name=run.label,
                marker_color=run.color,
                opacity=0.7,
            )
            # go.Bar(
            #     x=bar["centers"],
            #     y=bar["durations"],
            #     name=run.label,
            #     marker_color=run.color,
            #     opacity=0.7,
            #     width=bar["widths"],
            # )
        )

    fig.update_layout(
        # title="Overload magnitude histogram",
        xaxis_title="Overload magnitude [kW]",
        yaxis_title="Duration [s]",
        barmode="overlay",
    )
    _finalize_figure(
        fig,
        show=show,
        save=save,
        name=name or "overload_histogram",
        output_dir=output_dir,
    )
    return fig


def plot_tracking_error_histogram(
    runs: Sequence[RunData],
    *,
    bins: int | str = "auto",
    absolute: bool = True,
    show: bool = True,
    save: bool = False,
    name: str | None = None,
    output_dir: str | Path = "figures",
) -> go.Figure:
    fig = go.Figure()
    for run in runs:
        err = _tracking_error_kw(run.df)
        values = err.abs() if absolute else err

        bar = _build_histogram_bar(values, run.df, bins=bins)
        fig.add_trace(
            go.Scatter(
                x=bar["centers"],
                y=bar["durations"],
                mode="lines",
                name=run.label,
                marker_color=run.color,
                opacity=0.7,
            )
            # go.Bar(
            #     x=bar["centers"],
            #     y=bar["durations"],
            #     name=run.label,
            #     marker_color=run.color,
            #     opacity=0.7,
            #     width=bar["widths"],
            # )
        )

    fig.update_layout(
        # title="Tracking error magnitude histogram" if absolute else "Tracking error histogram",
        xaxis_title="Tracking error [kW]",
        yaxis_title="Duration [s]",
        barmode="overlay",
    )
    _finalize_figure(
        fig,
        show=show,
        save=save,
        name=name or ("tracking_error_magnitude" if absolute else "tracking_error"),
        output_dir=output_dir,
    )
    return fig


def plot_timeseries_power(
    runs: Sequence[RunData],
    *,
    time_unit: Literal["s", "min", "h"] = "s",
    show_total: bool = True,
    show_setpoint: bool = True,
    show_ev_consumption: bool = True,
    show_ev_setpoint: bool = False,
    split_power_subplots: bool = False,
    show: bool = True,
    save: bool = False,
    name: str | None = None,
    output_dir: str | Path = "figures",
    image_format: str = "pdf",
    sanitize_name: bool = True,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> go.Figure:
    seconds_per_unit = {"s": 1.0, "min": 60.0, "h": 3600.0}
    if time_unit not in seconds_per_unit:
        msg = f"time_unit must be one of {tuple(seconds_per_unit)}, got {time_unit!r}"
        raise ValueError(msg)
    time_scale = seconds_per_unit[time_unit]

    if split_power_subplots:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            row_heights=[0.58, 0.42],
            # subplot_titles=("P_cons and P_ref", "EV consumption"),
        )
    else:
        fig = go.Figure()

    # ev_colors = {"tesla": "#288e28", "vw": "#f68300", "skoda": "#4a4cd2"}
    ev_colors = _build_ev_color_map(runs)
    ev_line_width = 3.0
    for run in runs:
        x = _time_series_x(run.df) / time_scale
        if show_total and "p_total_kw" in run.df.columns:
            total_trace = go.Scatter(
                x=x,
                y=run.df["p_total_kw"],
                mode="lines",
                name="P_cons",
                line={"color": "blue"},
                legendrank=0,
            )
            if split_power_subplots:
                fig.add_trace(total_trace, row=1, col=1)
                fig.update_layout(yaxis_range=[120,180]) # change y-axis HERE
            else:
                fig.add_trace(total_trace)
        if show_setpoint and "p_central_setpoint_kw" in run.df.columns:
            setpoint_trace = go.Scatter(
                x=x,
                y=run.df["p_central_setpoint_kw"],
                mode="lines",
                name="P_ref",
                line={"color": "black", "dash": "dash"},
                # line_shape="hv",
                legendrank=1,
            )
            if split_power_subplots:
                fig.add_trace(setpoint_trace, row=1, col=1)
            else:
                fig.add_trace(setpoint_trace)

        if show_ev_consumption:
            for col in _select_columns_with_prefix(run.df, "p_cons_ev_"):
                ev_name = _ev_name_from_column(col, "p_cons_ev_")
                ev_trace = go.Scatter(
                    x=x,
                    y=run.df[col],
                    mode="lines",
                    name=ev_name,
                    line={"dash": "dot", "color": ev_colors.get(ev_name, run.color), "width": ev_line_width},
                    opacity=0.6,
                    legendrank=2,
                )
                if split_power_subplots:
                    fig.add_trace(ev_trace, row=2, col=1)
                else:
                    fig.add_trace(ev_trace)

        if show_ev_setpoint:
            for col in _select_columns_with_prefix(run.df, "p_mc_setpoint_ev_"):
                ev_name = _ev_name_from_column(col, "p_mc_setpoint_ev_")
                ev_setpoint_trace = go.Scatter(
                    x=x,
                    y=run.df[col],
                    mode="lines",
                    name=f"{ev_name} setpoint",
                    line={"dash": "dashdot", "color": ev_colors.get(ev_name, run.color), "width": ev_line_width},
                    opacity=0.6,
                    legendrank=3,
                )
                if split_power_subplots:
                    fig.add_trace(ev_setpoint_trace, row=2, col=1)
                else:
                    fig.add_trace(ev_setpoint_trace)

    if split_power_subplots:
        fig.update_layout(
            template="plotly_white",
            font={"family": "Times New Roman"},
            # title={"text": name or "power_timeseries", "font": {"family": "Times New Roman", "size": 35, "color": "black"}},
            legend={"font": {"family": "Times New Roman", "size": 35}},
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin={"l": 110, "r": 40, "t": 80, "b": 90},
            width=1600,
            height=900,
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=True,
            showline=True,
            linecolor="black",
            mirror=False,
            row=1,
            col=1,
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=True,
            showline=True,
            linecolor="black",
            mirror=False,
            row=2,
            col=1,
        )
        if x_range is not None:
            converted_x_range = [x_range[0] / time_scale, x_range[1] / time_scale]
            fig.update_xaxes(range=converted_x_range, row=1, col=1)
            fig.update_xaxes(range=converted_x_range, row=2, col=1)
        fig.update_yaxes(
            title_text="Power [kW]",
            title_font={"family": "Times New Roman", "size": 35, "color": "black"},
            tickfont={"size": 25, "color": "black"},
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=True,
            showline=True,
            linecolor="black",
            mirror=False,
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text="Power [kW]",
            title_font={"family": "Times New Roman", "size": 35, "color": "black"},
            tickfont={"size": 25, "color": "black"},
            tickmode="array",
            tickvals=[0, 2, 4, 6, 8, 10],
            range=[0, 11.5],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=True,
            showline=True,
            linecolor="black",
            mirror=False,
            row=2,
            col=1,
        )
        fig.update_xaxes(
            title_text=f"Time [{time_unit}]",
            title_font={"family": "Times New Roman", "size": 35, "color": "black"},
            tickfont={"size": 25, "color": "black"},
            automargin=True,
            row=2,
            col=1,
        )
    else:
        fig.update_layout(
            template="plotly_white",
            font={"family": "Times New Roman"},
            xaxis_title={"text": f"Time [{time_unit}]", "font": {"family": "Times New Roman", "size": 35, "color": "black"}},
            yaxis_title={"text": "Power [kW]", "font": {"family": "Times New Roman", "size": 35, "color": "black"}},
            xaxis={"tickfont": {"size": 25, "color": "black"}, "automargin": True},
            yaxis={"tickfont": {"size": 25, "color": "black"}, "automargin": True},
            legend={"font": {"family": "Times New Roman", "size": 35}},
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin={"l": 110, "r": 40, "t": 40, "b": 130},
            width=1600,
            height=600,
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=True,
            showline=True,
            linecolor="black",
            mirror=False,
        )
        if x_range is not None:
            fig.update_xaxes(range=[x_range[0] / time_scale, x_range[1] / time_scale])
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=True,
            showline=True,
            linecolor="black",
            mirror=False,
        )
        if y_range is not None:
            yaxis_kwargs = {"range": [y_range[0], y_range[1]]}
            if np.isclose(y_range[0], -0.5) and np.isclose(y_range[1], 24):
                yaxis_kwargs["dtick"] = 2
            fig.update_yaxes(**yaxis_kwargs)
    _finalize_figure(
        fig,
        show=show,
        save=save,
        name=name or "power_timeseries",
        output_dir=output_dir,
        image_format=image_format,
        sanitize_name=sanitize_name,
    )
    return fig


def plot_ev_efficiency_curves(
    ev_config_paths: dict[str, str | Path] | None = None,
    *,
    show: bool = True,
    save: bool = False,
    name: str | None = None,
    output_dir: str | Path = "figures",
    image_format: str = "pdf",
    sanitize_name: bool = True,
) -> go.Figure:
    ev_config_paths = ev_config_paths or EV_EFFICIENCY_CONFIGS
    fig = go.Figure()
    for ev_name, config_path in ev_config_paths.items():
        efficiency = _load_ev_efficiency(config_path)
        x = sorted(efficiency)
        y = [efficiency[current] for current in x]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=ev_name,
                line={"color": EV_COLORS.get(ev_name, qualitative.Plotly[0]), "width": 2.8},
                line_shape="hv",
                marker={"color": EV_COLORS.get(ev_name, qualitative.Plotly[0]), "size": 9},
            )
        )

    fig.update_layout(
        template="plotly_white",
        font={"family": "Times New Roman"},
        xaxis_title={"text": "Current [A]", "font": {"family": "Times New Roman", "size": 35, "color": "black"}},
        yaxis_title={"text": "Efficiency (η)", "font": {"family": "Times New Roman", "size": 35, "color": "black"}},
        xaxis={
            "tickfont": {"size": 25, "color": "black"},
            "tickmode": "array",
            "tickvals": [6, 8, 10, 12, 14, 16],
            "automargin": True,
        },
        yaxis={"tickfont": {"size": 25, "color": "black"}, "automargin": True},
        legend={"font": {"family": "Times New Roman", "size": 35}},
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 110, "r": 40, "t": 40, "b": 130},
        width=1600,
        height=600,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        zeroline=True,
        showline=True,
        linecolor="black",
        mirror=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        zeroline=True,
        showline=True,
        linecolor="black",
        mirror=False,
        range=[0.86, 0.905],
        tickmode="array",
        tickvals=[0.86, 0.87, 0.88, 0.89, 0.90],
    )
    _finalize_figure(
        fig,
        show=show,
        save=save,
        name=name or "ev_efficiency_curves",
        output_dir=output_dir,
        image_format=image_format,
        sanitize_name=sanitize_name,
    )
    return fig


def plot_timeseries_cons_priority_ratios(
    runs: Sequence[RunData],
    *,
    show: bool = True,
    save: bool = False,
    name: str | None = None,
    output_dir: str | Path = "figures",
) -> go.Figure:
    fig = go.Figure()
    ev_colors = _build_ev_color_map(runs)
    for run in runs:
        x = _time_series_x(run.df)
        consensus_dfs = _compute_consensus_dfs(run.df)
        cons_priority_ratios = consensus_dfs["cons_priority_ratios"]
        for ev_name in cons_priority_ratios.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=cons_priority_ratios[ev_name],
                    mode="lines",
                    name=f"{run.label} {ev_name}",
                    line={"color": ev_colors[ev_name]},
                    opacity=0.65,
                )
            )

    fig.update_layout(
        # title="Consumption-priority ratios over time",
        xaxis_title="Time [s]",
        yaxis_title="Consumption-priority ratio",
    )
    _finalize_figure(
        fig,
        show=show,
        save=save,
        name=name or "cons_priority_ratios_timeseries",
        output_dir=output_dir,
    )
    return fig


def plot_timeseries_consensus_gap(
    runs: Sequence[RunData],
    *,
    show: bool = True,
    save: bool = False,
    name: str | None = None,
    output_dir: str | Path = "figures",
) -> go.Figure:
    fig = go.Figure()
    for run in runs:
        x = _time_series_x(run.df)
        consensus_dfs = _compute_consensus_dfs(run.df)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=consensus_dfs["consensus_gap"],
                mode="lines",
                name=run.label,
                line={"color": run.color},
            )
        )

    fig.update_layout(
        # title="Consensus gap over time",
        xaxis_title="Time [s]",
        yaxis_title="Consensus gap",
    )
    _finalize_figure(
        fig,
        show=show,
        save=save,
        name=name or "consensus_gap_timeseries",
        output_dir=output_dir,
    )
    return fig


def plot_timeseries_consensus_mad(
    runs: Sequence[RunData],
    *,
    show: bool = True,
    save: bool = False,
    name: str | None = None,
    output_dir: str | Path = "figures",
) -> go.Figure:
    fig = go.Figure()
    for run in runs:
        x = _time_series_x(run.df)
        consensus_dfs = _compute_consensus_dfs(run.df)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=consensus_dfs["consensus_mad"],
                mode="lines",
                name=run.label,
                line={"color": run.color},
            )
        )

    fig.update_layout(
        # title="Consensus MAD over time",
        xaxis_title="Time [s]",
        yaxis_title="Consensus MAD",
    )
    _finalize_figure(
        fig,
        show=show,
        save=save,
        name=name or "consensus_mad_timeseries",
        output_dir=output_dir,
    )
    return fig


def plot_qq_consensus_mad(
    runs: Sequence[RunData],
    *,
    quantile_count: int = 101,
    show: bool = True,
    save: bool = False,
    name: str | None = None,
    output_dir: str | Path = "figures",
) -> go.Figure:
    if len(runs) != 2:
        msg = "Consensus MAD QQ-plot requires exactly two runs."
        raise ValueError(msg)

    x_run, y_run = runs
    x_values = _clean_numeric_array(_compute_consensus_dfs(x_run.df)["consensus_mad"])
    y_values = _clean_numeric_array(_compute_consensus_dfs(y_run.df)["consensus_mad"])
    if x_values.size == 0 or y_values.size == 0:
        msg = "Consensus MAD QQ-plot requires non-empty consensus MAD values for both runs."
        raise ValueError(msg)

    probabilities = np.linspace(0.0, 0.99, max(2, quantile_count))
    x_quantiles = np.nanquantile(x_values, probabilities)
    y_quantiles = np.nanquantile(y_values, probabilities)
    reference_min = float(min(x_quantiles.min(), y_quantiles.min()))
    reference_max = float(max(x_quantiles.max(), y_quantiles.max()))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_quantiles,
            y=y_quantiles,
            mode="markers",
            name=f"{y_run.label} vs {x_run.label}",
            marker={"color": y_run.color, "size": 7, "opacity": 0.75},
            customdata=probabilities,
            hovertemplate=(
                "Quantile: %{customdata:.2f}<br>"
                f"{x_run.label}: " + "%{x:.4f}<br>"
                f"{y_run.label}: " + "%{y:.4f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[reference_min, reference_max],
            y=[reference_min, reference_max],
            mode="lines",
            name="Equal quantiles",
            line={"color": "black", "dash": "dash"},
        )
    )

    fig.update_layout(
        # title="Consensus MAD QQ-plot",
        xaxis_title=f"{x_run.label} consensus MAD quantiles",
        yaxis_title=f"{y_run.label} consensus MAD quantiles",
    )
    _finalize_figure(
        fig,
        show=show,
        save=save,
        name=name or "consensus_mad_qq",
        output_dir=output_dir,
    )
    return fig


def plot_distribution_consensus_mad(
    runs: Sequence[RunData],
    *,
    point_count: int = 300,
    bandwidth: float | None = None,
    show: bool = True,
    save: bool = False,
    name: str | None = None,
    output_dir: str | Path = "figures",
) -> go.Figure:
    if len(runs) != 2:
        msg = "Consensus MAD distribution plot requires exactly two runs."
        raise ValueError(msg)

    fig = go.Figure()
    for run in runs:
        values = _clean_numeric_array(_compute_consensus_dfs(run.df)["consensus_mad"])
        if values.size == 0:
            msg = f"Consensus MAD distribution plot requires non-empty values for {run.label}."
            raise ValueError(msg)

        x, density = _estimate_pdf(values, point_count=point_count, bandwidth=bandwidth)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=density,
                mode="lines",
                name=run.label,
                line={"color": run.color},
            )
        )

    fig.update_layout(
        # title="Consensus MAD estimated probability density",
        xaxis_title="Consensus MAD",
        yaxis_title="Estimated probability density",
    )
    _finalize_figure(
        fig,
        show=show,
        save=save,
        name=name or "consensus_mad_distribution",
        output_dir=output_dir,
    )
    return fig


def export_figures(
    figures: dict[str, go.Figure],
    *,
    output_dir: str | Path,
    write_html: bool = True,
    write_image: bool = True,
    image_format: str = "png",
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, fig in figures.items():
        slug = _slugify(name)
        if write_html:
            path = output_path / f"{slug}.html"
            fig.write_html(path)
            written.append(path)
        if write_image:
            path = output_path / f"{slug}.{image_format}"
            fig.write_image(path)
            written.append(path)
    return written


def _finalize_figure(
    fig: go.Figure,
    *,
    show: bool,
    save: bool,
    name: str,
    output_dir: str | Path,
    image_format: str = "png",
    sanitize_name: bool = True,
) -> None:
    if show:
        fig.show()
    if save:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_name = _slugify(name) if sanitize_name else name
        fig.write_image(output_path / f"{file_name}.{image_format}")


def _time_series_x(df: pd.DataFrame) -> np.ndarray:
    if "t" in df.columns:
        return pd.to_numeric(df["t"], errors="coerce").to_numpy()
    if df.index.name == "t":
        return pd.to_numeric(df.index, errors="coerce").to_numpy()
    return np.arange(df.shape[0], dtype=float) * infer_dt_s(df)


def _load_ev_efficiency(config_path: str | Path) -> dict[float, float]:
    with Path(config_path).open() as config_file:
        config = yaml.safe_load(config_file) or {}
    efficiency = config.get("efficiency")
    if not isinstance(efficiency, dict):
        msg = f"Missing efficiency mapping in {config_path}"
        raise ValueError(msg)
    return {float(current): float(value) for current, value in efficiency.items()}


def _compute_consensus_dfs(df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    result = compute_metrics(df)
    if isinstance(result, tuple):
        consensus_dfs, _metrics = result
        return consensus_dfs

    return {
        "cons_priority_ratios": result.cons_priority_ratios,
        "consensus_gap": result.consensus_gap,
        "consensus_mad": result.consensus_mad,
    }


def _clean_numeric_array(values: pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
    clean = pd.to_numeric(pd.Series(np.ravel(values)), errors="coerce").dropna()
    return clean.to_numpy(dtype=float)


def _estimate_pdf(
    values: np.ndarray, *, point_count: int = 300, bandwidth: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return np.array([]), np.array([])

    if clean.size == 1 or float(clean.min()) == float(clean.max()):
        center = float(clean[0])
        width = max(abs(center) * 0.1, 1e-3)
        x = np.linspace(center - width, center + width, max(2, point_count))
        density = np.zeros_like(x)
        density[np.argmin(np.abs(x - center))] = 1.0
        return x, density

    resolved_bandwidth = bandwidth or _silverman_bandwidth(clean)
    lower = float(clean.min() - 3.0 * resolved_bandwidth)
    upper = float(clean.max() + 3.0 * resolved_bandwidth)
    x = np.linspace(lower, upper, max(2, point_count))
    z = (x[:, None] - clean[None, :]) / resolved_bandwidth
    density = np.exp(-0.5 * z**2).sum(axis=1)
    density /= clean.size * resolved_bandwidth * np.sqrt(2.0 * np.pi)
    return x, density


def _silverman_bandwidth(values: np.ndarray) -> float:
    std = float(np.std(values, ddof=1))
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    scale = min(std, iqr / 1.34) if iqr > 0 else std
    bandwidth = 0.9 * scale * values.size ** (-1 / 5)
    return max(bandwidth, 1e-6)


def _tracking_error_kw(df: pd.DataFrame) -> pd.Series:
    _validate_columns(df, {"p_total_kw", "p_central_setpoint_kw"})
    return df["p_total_kw"] - df["p_central_setpoint_kw"]


def _overload_magnitude_kw(df: pd.DataFrame) -> pd.Series:
    return _tracking_error_kw(df).clip(lower=0.0)


def _select_columns_with_prefix(df: pd.DataFrame, prefix: str) -> list[str]:
    return [col for col in df.columns if col.startswith(prefix)]


def _build_ev_color_map(runs: Sequence[RunData]) -> dict[str, str]:
    ev_names: list[str] = []
    for run in runs:
        for prefix in ("p_cons_ev_", "p_mc_setpoint_ev_", "rel_priority_ev_"):
            for col in _select_columns_with_prefix(run.df, prefix):
                ev_name = _ev_name_from_column(col, prefix)
                if ev_name not in ev_names:
                    ev_names.append(ev_name)

    colors = list(qualitative.Plotly)
    return {ev_name: colors[idx % len(colors)] for idx, ev_name in enumerate(ev_names)}


def _ev_name_from_column(col: str, prefix: str) -> str:
    return col.removeprefix(prefix)


def _resolve_bins(values: pd.Series | np.ndarray, bins: int | str) -> int | None:
    if isinstance(bins, int):
        return max(1, bins)
    if bins == "auto":
        clean = pd.to_numeric(values, errors="coerce")
        if isinstance(clean, pd.Series):
            clean = clean.dropna().to_numpy()
        else:
            clean = np.asarray(clean, dtype=float)
            clean = clean[~np.isnan(clean)]
        if clean.size == 0:
            return None
        return int(np.ceil(np.sqrt(clean.size)))
    return None


def _build_histogram_bar(
    values: pd.Series | np.ndarray, df: pd.DataFrame, *, bins: int | str
) -> dict[str, np.ndarray]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if clean.size == 0:
        return {
            "centers": np.array([]),
            "durations": np.array([]),
            "widths": np.array([]),
        }
    bin_count = _resolve_bins(clean, bins) or int(np.ceil(np.sqrt(clean.size)))
    counts, edges = np.histogram(clean, bins=bin_count)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = edges[1:] - edges[:-1]
    dt_s = infer_dt_s(df)
    durations = counts.astype(float) * dt_s
    return {
        "centers": centers,
        "durations": durations,
        "widths": widths,
    }


def _slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "figure"


def _validate_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(df.columns)
    if missing:
        msg = f"Missing required columns: {sorted(missing)}"
        raise ValueError(msg)


if __name__ == "__main__":
    figures_dir = Path(__file__).resolve().parents[3] / "figures" / "case_study"
    # figures_dir = Path(__file__).resolve().parents[3] / "figures" / "dev_tests"
    run_specs = [
        # RunSpec(path="results/test_default_1.csv", label="Default", color="blue"),
        # RunSpec(path="results/test_centralized.csv", label="Centralized", color="red"),
        # RunSpec(path="results/case_study/test_5_SC_30.csv", label="SC", color="blue"),
        RunSpec(path="results/case_study/test_5_Central.csv", label="central", color="red"),
        # RunSpec(path="results/dev_tests/test_1_sc.csv", label="central", color="red"),
        # RunSpec(path="results/practical_test/case_0.csv", label="central", color="red"),
    ]
    runs = load_runs(run_specs)
    save = False
    # plot_ev_efficiency_curves(
    #     show=True,
    #     save=save,
    #     output_dir=Path(__file__).resolve().parents[3] / "figures",
    #     image_format="pdf",
    # )
    plot_timeseries_power(
        runs,
        time_unit="min",
        show=True,
        save=save,
        name=Path(runs[0].path).stem if runs else None,
        output_dir=figures_dir,
        sanitize_name=False,
        split_power_subplots=True, # set True when there are over 10 EVs to get two subplots
        # y_range=(-0.5, 24),
        # x_range=(180, 250),
    )


# test_5_Central
# test_5_SC_30
# test_5_SC_100
