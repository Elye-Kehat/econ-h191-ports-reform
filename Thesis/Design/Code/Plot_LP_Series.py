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

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="Data/LP/LP_Panel.tsv")
    ap.add_argument("--outdir", default="Design/Output Data/visuals")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--ymax", type=float, default=120, help="Optional y-axis max for all plots")
    ap.add_argument("--width", type=float, default=9.5)
    ap.add_argument("--height", type=float, default=5.5)
    return ap.parse_args()

def to_datetime(row):
    if row["freq"] == "M":
        y = int(row["year"]); m = int(row["month"]) if pd.notna(row["month"]) else 1
        return pd.Timestamp(year=y, month=m, day=1)
    q = str(row["quarter"]).strip()
    if q not in Q_TO_MONTH:
        return pd.NaT
    y = int(row["year"]); m = Q_TO_MONTH[q]
    return pd.Timestamp(year=y, month=m, day=1)

def load_and_prep(panel_path: str) -> pd.DataFrame:
    df = pd.read_csv(panel_path, sep="\t")
    required = {"series_id","freq","port","year","LP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"LP_Panel missing columns: {missing}")
    df["date"] = df.apply(to_datetime, axis=1)
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values(["series_id","date"]).reset_index(drop=True)
    return df

def ensure_outdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def plot_series(ax, df, series_ids, title, ymax=None):
    for sid in series_ids:
        sub = df[df["series_id"] == sid]
        if sub.empty: 
            continue
        ax.plot(
            sub["date"], sub["LP"],
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
    plot_series(ax, df, ["Haifa_port_M", "Haifa_Legacy_Q", "Haifa_SIPG_Q"], "Haifa — LP over time", ymax=args.ymax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "haifa_lp.png"), dpi=args.dpi)
    plt.close(fig)

    # Ashdod
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    plot_series(ax, df, ["Ashdod_port_M", "Ashdod_Legacy_Q", "Ashdod_HCT_Q"], "Ashdod — LP over time", ymax=args.ymax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "ashdod_lp.png"), dpi=args.dpi)
    plt.close(fig)

    # All series
    fig, ax = plt.subplots(figsize=(args.width+1.5, args.height+0.5))
    plot_series(ax, df, SERIES_ORDER, "All Series — LP over time", ymax=args.ymax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "all_lp.png"), dpi=args.dpi)
    plt.close(fig)

    print(f"[viz] wrote plots to: {args.outdir}")

if __name__ == "__main__":
    main()
