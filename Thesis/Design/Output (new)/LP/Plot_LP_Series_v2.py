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

# v2 changes:
# 1. Entrant series are clipped to begin only after actual opening dates.
# 2. Port-total series are stitched so they run through the full sample:
#    prefer monthly port totals early, then append quarterly aggregate-port totals
#    when the monthly direct port series stops. If quarterly aggregate-port totals
#    are unavailable, fall back to computing a quarterly aggregate from terminal rows
#    when enough information is present in the panel.

PORT_MONTHLY_CANDIDATES = {
    "haifa_port": ["Haifa_port_M"],
    "ashdod_port": ["Ashdod_port_M"],
}

PORT_QUARTERLY_CANDIDATES = {
    "haifa_port": ["Haifa_port_Q"],
    "ashdod_port": ["Ashdod_port_Q"],
}

TERMINAL_CANDIDATES = {
    "haifa_legacy": ["Haifa_Legacy_Q"],
    "haifa_bayport": ["Haifa_SIPG_Q", "Haifa_Bayport_Q"],
    "ashdod_legacy": ["Ashdod_Legacy_Q"],
    "ashdod_hct": ["Ashdod_HCT_Q"],
}

TERMINAL_START = {
    "haifa_bayport": pd.Timestamp(2021, 9, 1),
    "ashdod_hct": pd.Timestamp(2022, 11, 1),
}

PORT_TERMINAL_COMPONENTS = {
    "haifa_port": ["haifa_legacy", "haifa_bayport"],
    "ashdod_port": ["ashdod_legacy", "ashdod_hct"],
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

NUMERATOR_CANDIDATES = [
    "tons",
    "throughput_tons",
    "throughput",
    "output",
    "Y",
    "y",
    "TEU",
    "teu",
]

DENOMINATOR_CANDIDATES = [
    "labor_hours",
    "hours",
    "L",
    "l",
    "labor",
    "hours_implied",
]


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


def resolve_series_id(df: pd.DataFrame, candidates: list[str]) -> str | None:
    present = set(df["series_id"].astype(str).unique())
    for cand in candidates:
        if cand in present:
            return cand
    return None


def get_series_by_id(df: pd.DataFrame, sid: str | None) -> pd.DataFrame:
    if sid is None:
        return pd.DataFrame(columns=df.columns)
    return df[df["series_id"] == sid].copy().sort_values("date").reset_index(drop=True)


def clip_terminal_start(sub: pd.DataFrame, logical_name: str) -> pd.DataFrame:
    if logical_name not in TERMINAL_START:
        return sub.copy()
    start_date = TERMINAL_START[logical_name]
    out = sub[sub["date"] >= start_date].copy()
    return out.sort_values("date").reset_index(drop=True)


def get_terminal_series(df: pd.DataFrame, logical_name: str) -> pd.DataFrame:
    sid = resolve_series_id(df, TERMINAL_CANDIDATES[logical_name])
    sub = get_series_by_id(df, sid)
    sub = clip_terminal_start(sub, logical_name)
    sub["logical_name"] = logical_name
    sub["source_type"] = "terminal"
    return sub


def pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            return col
    return None


def compute_quarterly_port_from_terminals(df: pd.DataFrame, port_logical_name: str) -> pd.DataFrame:
    components = PORT_TERMINAL_COMPONENTS[port_logical_name]
    pieces = []

    for logical_name in components:
        sub = get_terminal_series(df, logical_name)
        if not sub.empty:
            pieces.append(sub)

    if not pieces:
        return pd.DataFrame(columns=df.columns)

    big = pd.concat(pieces, ignore_index=True)
    num_col = pick_first_existing_column(big, NUMERATOR_CANDIDATES)
    den_col = pick_first_existing_column(big, DENOMINATOR_CANDIDATES)

    if num_col is None or den_col is None:
        return pd.DataFrame(columns=df.columns)

    big[num_col] = pd.to_numeric(big[num_col], errors="coerce")
    big[den_col] = pd.to_numeric(big[den_col], errors="coerce")
    big = big.dropna(subset=[num_col, den_col]).copy()

    if big.empty:
        return pd.DataFrame(columns=df.columns)

    agg = (
        big.groupby("date", as_index=False)[[num_col, den_col]]
        .sum()
        .rename(columns={num_col: "_num", den_col: "_den"})
    )
    agg = agg[agg["_den"] > 0].copy()
    if agg.empty:
        return pd.DataFrame(columns=df.columns)

    agg["LP"] = agg["_num"] / agg["_den"]
    agg["series_id"] = f"{port_logical_name}_computed_Q"
    agg["freq"] = "Q"
    agg["year"] = agg["date"].dt.year
    agg["quarter"] = "Q" + agg["date"].dt.quarter.astype(str)
    agg["logical_name"] = port_logical_name
    agg["source_type"] = "terminal_sum"

    # Keep useful raw columns if they can help later inspection.
    agg[num_col] = agg["_num"]
    agg[den_col] = agg["_den"]
    return agg


def build_port_total_series(df: pd.DataFrame, port_logical_name: str) -> pd.DataFrame:
    sid_month = resolve_series_id(df, PORT_MONTHLY_CANDIDATES[port_logical_name])
    sid_quarter = resolve_series_id(df, PORT_QUARTERLY_CANDIDATES[port_logical_name])

    monthly = get_series_by_id(df, sid_month)
    quarter = get_series_by_id(df, sid_quarter)

    if not monthly.empty:
        monthly["logical_name"] = port_logical_name
        monthly["source_type"] = "port_monthly"

    if not quarter.empty:
        quarter["logical_name"] = port_logical_name
        quarter["source_type"] = "port_quarterly"

    if quarter.empty:
        quarter = compute_quarterly_port_from_terminals(df, port_logical_name)

    pieces = []

    if not monthly.empty:
        pieces.append(monthly)

    if not quarter.empty:
        if not monthly.empty:
            quarter = quarter[quarter["date"] > monthly["date"].max()].copy()
        pieces.append(quarter)

    if not pieces:
        return pd.DataFrame(columns=df.columns)

    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
    return out


def get_plot_series(df: pd.DataFrame, logical_name: str) -> pd.DataFrame:
    if logical_name in ("haifa_port", "ashdod_port"):
        return build_port_total_series(df, logical_name)
    return get_terminal_series(df, logical_name)


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


def plot_one(ax: plt.Axes, sub: pd.DataFrame, logical_name: str) -> None:
    ax.plot(
        sub["date"],
        sub["LP"],
        linestyle=LINESTYLE[logical_name],
        linewidth=LINEWIDTH[logical_name],
        marker=MARKER[logical_name],
        markersize=MARKERSIZE[logical_name],
        label=SERIES_LABELS[logical_name],
    )


def plot_group(
    df: pd.DataFrame,
    logical_names: list[str],
    title: str,
    outpath: Path,
    events_key: str,
    legend_loc: str = "upper left",
) -> tuple[Path | None, dict]:
    fig, ax = plt.subplots(figsize=(12, 6))

    pieces = []
    plotted_any = False
    series_notes = {}

    for name in logical_names:
        sub = get_plot_series(df, name)
        if sub.empty:
            series_notes[name] = {"status": "missing"}
            continue

        plotted_any = True
        pieces.append(sub.copy())
        plot_one(ax, sub, name)

        series_notes[name] = {
            "status": "plotted",
            "first_date": str(sub["date"].min().date()),
            "last_date": str(sub["date"].max().date()),
            "n_obs": int(len(sub)),
            "source_types": sorted(sub.get("source_type", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
            "series_ids": sorted(sub["series_id"].dropna().astype(str).unique().tolist()),
        }

    if not plotted_any:
        plt.close(fig)
        return None, series_notes

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
    return outpath, series_notes


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parents[2]

    panel_candidates = [
        root / "Data" / "LP" / "LP_Panel_monthly.tsv",
        root / "Data" / "LP" / "LP_Panel.tsv",
    ]
    panel_path = find_first_existing(panel_candidates)

    out_dir = root / "Design" / "Output (new)" / "LP" / "Visuals_v2"
    ensure_dir(out_dir)

    df = load_lp_panel(panel_path)

    outputs = []
    notes = {}

    out, meta = plot_group(
        df,
        ["haifa_port", "haifa_legacy", "haifa_bayport"],
        "Haifa LP series",
        out_dir / "plot_lp_haifa_all_v2.png",
        "Haifa",
        legend_loc="upper left",
    )
    if out is not None:
        outputs.append(str(out))
    notes["haifa_all"] = meta

    out, meta = plot_group(
        df,
        ["ashdod_port", "ashdod_legacy", "ashdod_hct"],
        "Ashdod LP series",
        out_dir / "plot_lp_ashdod_all_v2.png",
        "Ashdod",
        legend_loc="upper left",
    )
    if out is not None:
        outputs.append(str(out))
    notes["ashdod_all"] = meta

    out, meta = plot_group(
        df,
        ["haifa_port", "haifa_legacy", "haifa_bayport", "ashdod_port", "ashdod_legacy", "ashdod_hct"],
        "All main LP series",
        out_dir / "plot_lp_all_series_v2.png",
        "All",
        legend_loc="upper left",
    )
    if out is not None:
        outputs.append(str(out))
    notes["all_series"] = meta

    manifest = {
        "version": "v2",
        "sample_start": SAMPLE_START,
        "sample_end": SAMPLE_END,
        "panel_source": str(panel_path),
        "output_dir": str(out_dir),
        "outputs": outputs,
        "changes": [
            "Clipped Haifa-Bayport to start at 2021-09-01",
            "Clipped Ashdod-HCT to start at 2022-11-01",
            "Stitched port totals to continue beyond monthly direct-port coverage",
            "Fallback to computed quarterly aggregate port total from terminal rows when needed",
        ],
        "series_notes": notes,
    }

    with open(out_dir / "lp_visual_manifest_v2.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print(f"Source panel: {panel_path}")
    print(f"Output dir:   {out_dir}")
    for item in outputs:
        print(item)


if __name__ == "__main__":
    main()
