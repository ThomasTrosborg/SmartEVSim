from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly
import plotly.graph_objects as go
from smartevsim.utils.conversions import convert_A_to_kw
plotly.io.templates.default = "plotly_white"

tesla_color = "#2ca02c"
vw_color = "#ff7f0e"
plot_output_dir = Path(__file__).resolve().parents[1] / "figures" / "practical_tests"
split_intervals = [
    ("08:47:20", "08:52"),
    ("10:09:10", "10:14"),
    ("11:15:45", "11:20"),
    ("12:23:45", "12:26:30"),
    ("12:20:15", "12:23:30"),
]
buffer_seconds = 5

def load_data(
        file_path: str,
        ) -> pd.DataFrame: 

    # read the CSV file, using the timestamp and charger columns
    df = pd.read_csv(
        file_path,
        sep=";",
        usecols=lambda column: column == "ts" or column.startswith("Charger_"),
    )

    # convert the timestamp column to datetime format and set it as the index
    timestamps = df["ts"].astype(str).str.replace(
        r"(\d{2}/\d{2}/\d{4}) (\d{2})\.(\d{2})$",
        r"\1 \2:\3",
        regex=True,
    )
    df.index = pd.to_datetime(timestamps, dayfirst=True, format="mixed")
    df = df.sort_index()

    charger_columns = [column for column in df.columns if column.startswith("Charger_")]
    charger_labels = [column.split(" |")[0] for column in charger_columns]
    power_pattern = r'"System Power Psum \(W\)":([^,}]+)'
    setpoint_pattern = r'"Setpoint Implemented \(A\)":([^,}]+)'

    result = pd.DataFrame(index=df.index)

    # convert the charger columns to numeric values, handling missing or malformed data
    for source_column, label in zip(charger_columns, charger_labels):
        cleaned = (
            df[source_column]
            .fillna("")
            .astype(str)
            .str.strip('"')
            .str.replace('""', '"', regex=False)
        )
        result[label] = pd.to_numeric(
            cleaned.str.extract(power_pattern)[0],
            errors="coerce",
        )
        result[f"{label}_setpoint"] = pd.to_numeric(
            cleaned.str.extract(setpoint_pattern)[0],
            errors="coerce",
        ).apply(convert_A_to_kw)

    return result


def _clock_to_timestamp(reference_date: pd.Timestamp, clock_time: str) -> pd.Timestamp:
    parts = [int(part) for part in clock_time.split(":")]
    if len(parts) == 2:
        hours, minutes = parts
        seconds = 0
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Unsupported clock time format: {clock_time}")
    return reference_date.normalize() + timedelta(hours=hours, minutes=minutes, seconds=seconds)


def _build_power_plot_frame(
        df: pd.DataFrame,
        start_time: str,
        end_time: str,
        buffer_s: int = buffer_seconds,
        ) -> pd.DataFrame:
    reference_date = df.index.min()
    start_timestamp = _clock_to_timestamp(reference_date, start_time)
    end_timestamp = _clock_to_timestamp(reference_date, end_time)
    window_start = start_timestamp - timedelta(seconds=buffer_s)

    window = df.loc[(df.index >= start_timestamp) & (df.index <= end_timestamp)].copy()
    window.index = (window.index - window_start).total_seconds()
    window.index.name = "time_s"

    padding_index = pd.Index([0.0], name="time_s")
    padding = pd.DataFrame(0.0, index=padding_index, columns=df.columns)

    return pd.concat([window, padding]).sort_index()


def _add_power_traces(
        fig: go.Figure,
        df: pd.DataFrame,
        tesla_color: str = tesla_color,
        vw_color: str = vw_color,
        ) -> go.Figure:


    tesla_kw = df["Charger_1"].loc[~df["Charger_1"].isna()] / 1000
    vw_kw = df["Charger_2"].loc[~df["Charger_2"].isna()] / 1000
    tesla_setpoint = df["Charger_1_setpoint"].loc[~df["Charger_1_setpoint"].isna()]
    vw_setpoint = df["Charger_2_setpoint"].loc[~df["Charger_2_setpoint"].isna()]
    p_cons_kw = df[["Charger_1", "Charger_2"]].ffill().fillna(0).sum(axis=1) / 1000

    if not tesla_setpoint.empty:
        tesla_setpoint = pd.concat(
            [
                pd.Series([0.0], index=[df.index[0]]),
                tesla_setpoint,
            ]
        )
    if not vw_setpoint.empty:
        vw_setpoint = pd.concat(
            [
                pd.Series([0.0], index=[df.index[0]]),
                vw_setpoint,
            ]
        )

    # # or forward fill and then fill remaining NaNs with 0: 
    # tesla_kw = df["Charger_1"].ffill() / 1000
    # vw_kw = df["Charger_2"].ffill() / 1000
    # tesla_setpoint = df["Charger_1_setpoint"].ffill()
    # vw_setpoint = df["Charger_2_setpoint"].ffill()
    # p_cons_kw = df[["Charger_1", "Charger_2"]].ffill().fillna(0).sum(axis=1) / 1000

    fig.add_trace(go.Scatter(x=df.index, y=[15] * len(df), mode="lines", name="P_ref", line=dict(color="black", dash="dash"), legendrank=1,))
    fig.add_trace(go.Scatter(x=tesla_kw.index, y=tesla_kw, mode="lines", name="tesla", line=dict(color=tesla_color), legendrank=2,))
    fig.add_trace(go.Scatter(x=tesla_setpoint.index, y=tesla_setpoint, mode="lines", name="tesla setpoint", line=dict(color=tesla_color, dash="dash"), line_shape="hv", legendrank=3,))
    fig.add_trace(go.Scatter(x=vw_kw.index, y=vw_kw, mode="lines", name="vw", line=dict(color=vw_color), legendrank=4,))
    fig.add_trace(go.Scatter(x=vw_setpoint.index, y=vw_setpoint, mode="lines", name="vw setpoint", line=dict(color=vw_color, dash="dash"), line_shape="hv", legendrank=5,))
    fig.add_trace(go.Scatter(x=df.index, y=p_cons_kw, mode="lines", name="P_cons", line=dict(color="blue"), legendrank=0,))

    return fig


def _style_power_figure(
        fig: go.Figure,
        x_axis_title: str,
        x_tickformat: str | None = None,
        ) -> go.Figure:
    fig.update_layout(
        font={"family": "Times New Roman"},
        xaxis_title={"text": x_axis_title, "font": {"family": "Times New Roman", "size": 35, "color": "black"}},
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
        showgrid=True, gridcolor="rgba(0,0,0,0.15)", zeroline=True,
        showline=True, linecolor="black", mirror=False,
    )
    if x_tickformat is not None:
        fig.update_xaxes(tickformat=x_tickformat)
    fig.update_yaxes(
        range=[-0.5, 17],
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        zeroline=True,
        showline=True,
        linecolor="black",
        mirror=False,
    )
    return fig


def plot_power(
        df: pd.DataFrame, 
        title: str = "Power Consumption and Setpoint",
        tesla_color: str = tesla_color,
        vw_color: str = vw_color,):
    
    fig = go.Figure()
    _add_power_traces(fig, df, tesla_color=tesla_color, vw_color=vw_color)
    _style_power_figure(fig, x_axis_title="Time [h:m:s]", x_tickformat="%H:%M:%S")
    fig.show()


def plot_power_sections(
        df: pd.DataFrame,
        intervals: list[tuple[str, str]] = split_intervals,
        output_dir: str | Path = plot_output_dir,
        buffer_s: int = buffer_seconds,
        tesla_color: str = tesla_color,
        vw_color: str = vw_color,
        show: bool = False,
        ) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    for start_time, end_time in intervals:
        plot_frame = _build_power_plot_frame(
            df,
            start_time=start_time,
            end_time=end_time,
            buffer_s=buffer_s,
        )
        fig = go.Figure()
        _add_power_traces(fig, plot_frame, tesla_color=tesla_color, vw_color=vw_color)
        _style_power_figure(fig, x_axis_title="Time [s]")
        if show:
            fig.show()

        safe_start = start_time.replace(":", "-")
        safe_end = end_time.replace(":", "-")
        file_path = output_path / f"power_{safe_start}_{safe_end}.pdf"
        fig.write_image(file_path)
        written_files.append(file_path)

    return written_files




if __name__ == "__main__":
    df = load_data("practical_tests/Energydata export 02-07-2026 19-00-40.csv")
    plot_power(df)
    plot_power_sections(df)
