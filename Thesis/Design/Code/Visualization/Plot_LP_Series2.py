#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot_LP_Series.py
-----------------
Make publication-ready line plots for the six LP time series built in LP_Panel.tsv.

Outputs (default):
  Output Data/visuals/haifa_lp.png
  Output Data/visuals/ashdod_lp.png
  Output Data/visuals/all_lp.png

Usage example:
  python "Design/Code/Plot_LP_Series.py" \
    --panel "Data/LP/LP_Panel.tsv" \
    --outdir "Design/Output Data/visuals" \
    --dpi 220 --ymax 300
"""
import argparse, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FixedFormatter
import calendar


Q_TO_MONTH = {"Q1": 3, "Q2": 6, "Q3": 9, "Q4": 12}

SERIES_ORDER = [
    "Haifa_port_M",
    "Haifa_Legacy_Q",
    "Haifa_SIPG_Q",
    "Ashdod_port_M",
    "Ashdod_Legacy_Q",
    "Ashdod_HCT_Q",
]

LABELS = {
    "Haifa_port_M": "Haifa (Port, Monthly)",
    "Haifa_Legacy_Q": "Haifa-Legacy (Quarterly)",
    "Haifa_SIPG_Q": "Haifa-Bayport (Quarterly)",
    "Ashdod_port_M": "Ashdod (Port, Monthly)",
    "Ashdod_Legacy_Q": "Ashdod-Legacy (Quarterly)",
    "Ashdod_HCT_Q": "Ashdod-HCT (Quarterly)",
}

LINESTYLE = {
    "Haifa_port_M": "-",
    "Haifa_Legacy_Q": "--",
    "Haifa_SIPG_Q": "--",
    "Ashdod_port_M": "-",
    "Ashdod_Legacy_Q": "--",
    "Ashdod_HCT_Q": "--",
}

MARKERS = {
    "Haifa_port_M": "",
    "Haifa_Legacy_Q": "o",
    "Haifa_SIPG_Q": "s",
    "Ashdod_port_M": "",
    "Ashdod_Legacy_Q": "o",
    "Ashdod_HCT_Q": "s",
}


REFORM_EVENTS = {
    # Haifa-specific shocks
    "Haifa": [
        # Haifa Port competition entry: Bayport (SIPG) opens, 09-2021
        {"date": pd.Timestamp(2021, 9, 1), "label": "Bayport opens (Haifa)"},
        # Haifa-Legacy privatization, 01-2023
        {"date": pd.Timestamp(2023, 1, 1), "label": "Haifa privatized"},
    ],
    # Ashdod-specific shocks
    "Ashdod": [
        # Ashdod Port competition entry: HCT (Southport/TIL) effectively online, 11-2022
        {"date": pd.Timestamp(2022, 11, 1), "label": "HCT opens (Ashdod)"},
    ],
    # All-series plot gets the union of the above
    "All": [
        {"date": pd.Timestamp(2021, 9, 1), "label": "Bayport opens (Haifa)"},
        {"date": pd.Timestamp(2022, 11, 1), "label": "HCT opens (Ashdod)"},
        {"date": pd.Timestamp(2023, 1, 1), "label": "Haifa privatized"},
    ],
}


def add_reform_lines(ax, events):
    """
    Add vertical dashed lines and labels for port reform events.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to annotate.
    events : list of dict
        Each dict has keys 'date' (pd.Timestamp) and 'label' (str).
    """
    if not events:
        return

    # Use current y-limits so labels sit inside the plot area
    ymin, ymax = ax.get_ylim()
    text_y = ymax * 0.97

    for ev in events:
        dt = ev["date"]
        label = ev["label"]

        # Vertical dashed line at the event date
        ax.axvline(
            dt,
            linestyle="--",
            linewidth=1.0,
            color="k",
            alpha=0.7,
        )

        # Short vertical label just inside the top of the plot
        ax.text(
            dt,
            text_y,
            label,
            rotation=90,
            va="top",
            ha="right",
            fontsize=12,
        )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="Data/LP/LP_Panel.tsv")
    ap.add_argument("--outdir", default="Design/Output Data/visuals")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--ymax", type=float, default=120, help="Optional y-axis max for all plots")
    ap.add_argument("--width", type=float, default=9.5)
    ap.add_argument("--height", type=float, default=5.5)
    return ap.parse_args()


def configure_quarter_ticks_eoq(ax, df):
    # Major ticks: Jan 1 each year
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Compute range
    dmin = pd.to_datetime(df['date'].min()).normalize()
    dmax = pd.to_datetime(df['date'].max()).normalize()
    start_year = dmin.year
    end_year   = dmax.year

    # Minor ticks: end-of-quarter for Q1/Q2/Q3 = Mar 31, Jun 30, Sep 30
    minors = []
    labels = []
    for y in range(start_year, end_year + 1):
        for m, qlab in [(3, 'Q1'), (6, 'Q2'), (9, 'Q3')]:
            last_day = calendar.monthrange(y, m)[1]
            dt = pd.Timestamp(year=y, month=m, day=last_day)
            minors.append(mdates.date2num(dt))
            labels.append(qlab)

    ax.xaxis.set_minor_locator(FixedLocator(minors))
    ax.xaxis.set_minor_formatter(FixedFormatter(labels))

    ax.tick_params(axis='x', which='major', length=6)
    ax.tick_params(axis='x', which='minor', length=3, labelsize=8, pad=2)

    # Light guide lines
    ax.grid(True, which='major', axis='x', alpha=0.15)
    ax.grid(True, which='minor', axis='x', alpha=0.08)

    # Nice full-year bounds: [Jan 1, next Jan 1)
    ax.set_xlim(pd.Timestamp(start_year, 1, 1), pd.Timestamp(end_year + 1, 1, 1))


def configure_quarter_ticks(ax, df):
    # Major ticks: each Jan 1
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Build equal-thirds minor ticks between consecutive Jan 1's
    dmin = pd.to_datetime(df["date"].min())
    dmax = pd.to_datetime(df["date"].max())
    start_year = dmin.year
    end_year = dmax.year

    minors = []
    labels = []
    for year in range(start_year, end_year + 1):
        jan1 = pd.Timestamp(year=year, month=1, day=1)
        next_jan1 = pd.Timestamp(year=year + 1, month=1, day=1)
        # Place Q1/Q2/Q3 ticks evenly between Jan1 and next Jan1
        delta = (next_jan1 - jan1) / 4
        for q, qlab in enumerate(["Q1", "Q2", "Q3"], start=1):
            tick_date = jan1 + delta * q
            minors.append(mdates.date2num(tick_date))
            labels.append(qlab)

    ax.xaxis.set_minor_locator(FixedLocator(minors))
    ax.xaxis.set_minor_formatter(FixedFormatter(labels))

    ax.tick_params(axis="x", which="major", length=6)
    ax.tick_params(axis="x", which="minor", length=3, labelsize=8, pad=2)

    ax.grid(True, which="major", axis="x", alpha=0.15)
    ax.grid(True, which="minor", axis="x", alpha=0.08)

    ax.set_xlim(pd.Timestamp(start_year, 1, 1), pd.Timestamp(end_year + 1, 1, 1))


def to_datetime(row):
    """
    Convert (year, freq, maybe quarter/month) into a pandas Timestamp.
    Expected df columns: 'year', 'freq', and either 'month' or 'quarter'.
    """
    year = int(row["year"])
    freq = row["freq"]

    # Monthly series: use last day of the month (for nice alignment)
    if freq == "M":
        month = int(row["month"])
        last_day = calendar.monthrange(year, month)[1]
        return pd.Timestamp(year=year, month=month, day=last_day)

    # Quarterly series: place at end-of-quarter
    if freq == "Q":
        qstr = str(row.get("quarter", "Q1"))
        # handle values like 'Q1', 'Q2', etc.
        if isinstance(qstr, str) and qstr.upper().startswith("Q"):
            q = qstr.upper()
        else:
            # allow integers 1-4
            q = f"Q{int(qstr)}"
        month = Q_TO_MONTH[q]
        last_day = calendar.monthrange(year, month)[1]
        return pd.Timestamp(year=year, month=month, day=last_day)

    # Fallback: just use Jan 1 of the year
    return pd.Timestamp(year=year, month=1, day=1)


def load_and_prep(panel_path: str) -> pd.DataFrame:
    """
    Load LP_Panel.tsv and prepare for plotting.

    Expected schema (at minimum):
      - series_id: str (e.g., 'Haifa_port_M')
      - freq: 'M' or 'Q'
      - port: 'Haifa' or 'Ashdod'
      - year: int
      - LP: float
      - month or quarter: used to build a Timestamp
    """
    df = pd.read_csv(panel_path, sep="\t")

    # Basic checks
    required = {"series_id", "freq", "port", "year", "LP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"LP_Panel missing columns: {missing}")

    df["date"] = df.apply(to_datetime, axis=1)
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    return df


def ensure_outdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_series(ax, df, series_ids, title, ymax=None):
    for sid in series_ids:
        sub = df[df["series_id"] == sid]
        if sub.empty:
            continue
        ax.plot(
            sub["date"],
            sub["LP"],
            linestyle=LINESTYLE.get(sid, "-"),
            marker=MARKERS.get(sid, ""),
            linewidth=2.0,
            label=LABELS.get(sid, sid),
        )
    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("LP (mix-adjusted)")
    ax.grid(True, alpha=0.25)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.legend(frameon=False, ncols=1)


def main():
    args = parse_args()
    ensure_outdir(args.outdir)
    df = load_and_prep(args.panel)

    # Haifa
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    plot_series(
        ax,
        df,
        ["Haifa_port_M", "Haifa_Legacy_Q", "Haifa_SIPG_Q"],
        "Haifa — LP over time",
        ymax=args.ymax,
    )
    configure_quarter_ticks_eoq(ax, df)
    add_reform_lines(ax, REFORM_EVENTS["Haifa"])
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "haifa_lp.png"), dpi=args.dpi)
    plt.close(fig)

    # Ashdod
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    plot_series(
        ax,
        df,
        ["Ashdod_port_M", "Ashdod_Legacy_Q", "Ashdod_HCT_Q"],
        "Ashdod — LP over time",
        ymax=args.ymax,
    )
    configure_quarter_ticks_eoq(ax, df)
    add_reform_lines(ax, REFORM_EVENTS["Ashdod"])
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "ashdod_lp.png"), dpi=args.dpi)
    plt.close(fig)

    # All series
    fig, ax = plt.subplots(figsize=(args.width + 1.5, args.height + 0.5))
    plot_series(
        ax,
        df,
        SERIES_ORDER,
        "All Series — LP over time",
        ymax=args.ymax,
    )
    configure_quarter_ticks_eoq(ax, df)
    add_reform_lines(ax, REFORM_EVENTS["All"])
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "all_lp.png"), dpi=args.dpi)
    plt.close(fig)

    print(f"[viz] wrote plots to: {args.outdir}")


if __name__ == "__main__":
    main()
