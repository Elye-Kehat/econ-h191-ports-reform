#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import calendar
import json
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


SAMPLE_START = "2018-01"
SAMPLE_END = "2024-12"

# For the descriptive raw-series plot, prefer MONTHLY port totals so the figure
# starts at the full sample start, while keeping quarterly terminal series.
SERIES_CANDIDATES = {
    "haifa_port": ["Haifa_port_M", "Haifa_port_Q"],
    "haifa_legacy": ["Haifa_Legacy_Q"],
    "haifa_bayport": ["Haifa_SIPG_Q", "Haifa_Bayport_Q"],
    "ashdod_port": ["Ashdod_port_M", "Ashdod_port_Q"],
    "ashdod_legacy": ["Ashdod_Legacy_Q"],
    "ashdod_hct": ["Ashdod_HCT_Q"],
}

SERIES_LABELS = {
    "haifa_port": "Haifa total",
    "haifa_legacy": "Haifa-Legacy",
    "haifa_bayport": "Haifa-Bayport",
    "ashdod_port": "Ashdod total",
    "ashdod_legacy": "Ashdod-Legacy",
    "ashdod_hct": "Ashdod-HCT",
}

LINESTYLE = {
    "haifa_port": "-",
    "haifa_legacy": "--",
    "haifa_bayport": "--",
    "ashdod_port": "-",
    "ashdod_legacy": "--",
    "ashdod_hct": "--",
}

LINEWIDTH = {
    "haifa_port": 2.4,
    "haifa_legacy": 2.2,
    "haifa_bayport": 2.2,
    "ashdod_port": 2.4,
    "ashdod_legacy": 2.2,
    "ashdod_hct": 2.2,
}

# Add back markers. Keep port markers smaller so they do not overwhelm the plot.
MARKER = {
    "haifa_port": "o",
    "haifa_legacy": "o",
    "haifa_bayport": "s",
    "ashdod_port": "o",
    "ashdod_legacy": "o",
    "ashdod_hct": "s",
}

MARKERSIZE = {
    "haifa_port": 3.0,
    "haifa_legacy": 5.0,
    "haifa_bayport": 5.0,
    "ashdod_port": 3.0,
    "ashdod_legacy": 5.0,
    "ashdod_hct": 5.0,
}

REFORM_EVENTS = {
    "Haifa": [
        {"date": pd.Timestamp(2021, 9, 1), "label": "Bayport opens (Haifa)"},
        {"date": pd.Timestamp(2023, 1, 1), "label": "Haifa privatized"},
    ],
    "Ashdod": [
        {"date": pd.Timestamp(2022, 11, 1), "label": "HCT opens (Ashdod)"},
    ],
    "All": [
        {"date": pd.Timestamp(2021, 9, 1), "label": "Bayport opens (Haifa)"},
        {"date": pd.Timestamp(2022, 11, 1), "label": "HCT opens (Ashdod)"},
        {"date": pd.Timestamp(2023, 1, 1), "label": "Haifa privatized"},
    ],
}


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def find_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find any of these files:\n" + "\n".join(str(p) for p in paths)
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def month_end_timestamp(year: int, month: int) -> pd.Timestamp:
    last_day = calendar.monthrange(int(year), int(month))[1]
    return pd.Timestamp(int(year), int(month), last_day)


def quarter_end_timestamp(year: int, quarter: str) -> pd.Timestamp:
    q = str(quarter).strip().upper()
    q_to_month = {"Q1": 3, "Q2": 6, "Q3": 9, "Q4": 12}
    if q not in q_to_month:
        raise ValueError(f"Bad quarter value: {quarter}")
    return month_end_timestamp(year, q_to_month[q])


def resolve_series_id(df: pd.DataFrame, logical_name: str) -> str | None:
    present = set(df["series_id"].astype(str).unique())
    for cand in SERIES_CANDIDATES[logical_name]:
        if cand in present:
            return cand
    return None


def load_lp_panel(panel_path: Path) -> pd.DataFrame:
    df = read_tsv(panel_path).copy()

    required = {"series_id", "year", "LP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"LP panel missing columns: {sorted(missing)}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["LP"] = pd.to_numeric(df["LP"], errors="coerce")
    df = df.dropna(subset=["year", "LP"]).copy()
    df["year"] = df["year"].astype(int)

    date_vals = []
    for _, row in df.iterrows():
        row_freq = str(row.get("freq", "M")).upper()

        if row_freq == "Q":
            quarter = row.get("quarter", None)
            if pd.isna(quarter) or quarter is None:
                month = row.get("month", None)
                if pd.isna(month):
                    raise ValueError("Quarterly row missing both quarter and month.")
                qnum = ((int(month) - 1) // 3) + 1
                quarter = f"Q{qnum}"
            date_vals.append(quarter_end_timestamp(int(row["year"]), quarter))
        else:
            month = row.get("month", None)
            if pd.isna(month):
                raise ValueError("Monthly row missing month.")
            date_vals.append(month_end_timestamp(int(row["year"]), int(month)))

    df["date"] = date_vals
    df["date_str"] = df["date"].dt.strftime("%Y-%m")
    df = df[(df["date_str"] >= SAMPLE_START) & (df["date_str"] <= SAMPLE_END)].copy()
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    return df


def configure_month_ticks(ax: plt.Axes) -> None:
    start_ts = pd.Timestamp(f"{SAMPLE_START}-01")
    end_ts = pd.Timestamp(f"{SAMPLE_END}-01") + pd.offsets.MonthEnd(0)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

    ax.tick_params(axis="x", which="major", length=6)
    ax.tick_params(axis="x", which="minor", length=3, labelsize=8, pad=2)

    ax.grid(True, which="major", axis="x", alpha=0.15)
    ax.grid(True, which="minor", axis="x", alpha=0.08)

    ax.set_xlim(start_ts, pd.Timestamp(end_ts.year + 1, 1, 1))


def add_reform_lines(ax: plt.Axes, events: list[dict]) -> None:
    if not events:
        return

    ymin, ymax = ax.get_ylim()
    text_y = ymax * 0.97 if ymax > 0 else 1.0

    for ev in events:
        dt = ev["date"]
        label = ev["label"]
        ax.axvline(dt, linestyle="--", linewidth=1.0, color="k", alpha=0.7)
        ax.text(dt, text_y, label, rotation=90, va="top", ha="right", fontsize=11)


def style_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Date")
    ax.set_ylabel("Labor productivity (LP)")
    ax.grid(True, alpha=0.25)


def set_auto_ylim(ax: plt.Axes, combined_df: pd.DataFrame) -> None:
    ymax = pd.to_numeric(combined_df["LP"], errors="coerce").max()
    if pd.isna(ymax):
        return
    upper = max(10, math.ceil((float(ymax) * 1.10) / 10.0) * 10)
    ax.set_ylim(0, upper)


def plot_series(ax: plt.Axes, df: pd.DataFrame, logical_name: str) -> bool:
    sid = resolve_series_id(df, logical_name)
    if sid is None:
        return False

    sub = df[df["series_id"] == sid].copy().sort_values("date")
    if sub.empty:
        return False

    ax.plot(
        sub["date"],
        sub["LP"],
        linestyle=LINESTYLE[logical_name],
        linewidth=LINEWIDTH[logical_name],
        marker=MARKER[logical_name],
        markersize=MARKERSIZE[logical_name],
        label=SERIES_LABELS[logical_name],
    )
    return True


def plot_group(
    df: pd.DataFrame,
    logical_names: list[str],
    title: str,
    outpath: Path,
    events_key: str,
    legend_loc: str = "upper left",
) -> Path | None:
    fig, ax = plt.subplots(figsize=(12, 6))

    pieces = []
    plotted_any = False

    for name in logical_names:
        sid = resolve_series_id(df, name)
        if sid is None:
            continue
        sub = df[df["series_id"] == sid].copy().sort_values("date")
        if sub.empty:
            continue

        plotted_any = True
        pieces.append(sub)
        plot_series(ax, df, name)

    if not plotted_any:
        plt.close(fig)
        return None

    combined = pd.concat(pieces, ignore_index=True)

    ax.set_title(title, loc="left", fontsize=13, weight="bold")
    style_axis(ax)
    configure_month_ticks(ax)
    set_auto_ylim(ax, combined)
    add_reform_lines(ax, REFORM_EVENTS[events_key])
    ax.legend(frameon=False, loc=legend_loc)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return outpath


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parents[2]

    panel_candidates = [
        root / "Data" / "LP" / "LP_Panel_monthly.tsv",
        root / "Data" / "LP" / "LP_Panel.tsv",
    ]
    panel_path = find_first_existing(panel_candidates)

    out_dir = root / "Design" / "Output (new)" / "LP" / "Visuals"
    ensure_dir(out_dir)

    df = load_lp_panel(panel_path)

    outputs = []

    outputs.append(
        plot_group(
            df,
            ["haifa_port", "haifa_legacy", "haifa_bayport"],
            "Haifa LP series",
            out_dir / "plot_lp_haifa_all.png",
            "Haifa",
            legend_loc="upper left",
        )
    )

    outputs.append(
        plot_group(
            df,
            ["ashdod_port", "ashdod_legacy", "ashdod_hct"],
            "Ashdod LP series",
            out_dir / "plot_lp_ashdod_all.png",
            "Ashdod",
            legend_loc="upper left",
        )
    )

    outputs.append(
        plot_group(
            df,
            ["haifa_port", "haifa_legacy", "haifa_bayport", "ashdod_port", "ashdod_legacy", "ashdod_hct"],
            "All main LP series",
            out_dir / "plot_lp_all_series.png",
            "All",
            legend_loc="upper left",
        )
    )

    outputs = [str(p) for p in outputs if p is not None]

    manifest = {
        "sample_start": SAMPLE_START,
        "sample_end": SAMPLE_END,
        "panel_source": str(panel_path),
        "output_dir": str(out_dir),
        "outputs": outputs,
    }

    with open(out_dir / "lp_visual_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print(f"Source panel: {panel_path}")
    print(f"Output dir:   {out_dir}")
    for item in outputs:
        print(item)


if __name__ == "__main__":
    main()