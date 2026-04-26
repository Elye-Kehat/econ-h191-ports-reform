#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import calendar
import json
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


SAMPLE_START = "2018-01"
SAMPLE_END = "2024-12"

SERIES_META = {
    "Haifa_Legacy_Q": {
        "label": "Haifa-Legacy",
        "group": "Haifa",
        "kind": "legacy",
        "title": "Monthly LP: Haifa-Legacy",
    },
    "Haifa_SIPG_Q": {
        "label": "Haifa-Bayport",
        "group": "Haifa",
        "kind": "entrant",
        "title": "Monthly LP: Haifa Entrant (Bayport)",
    },
    "Haifa_port_M": {
        "label": "Haifa total",
        "group": "Haifa",
        "kind": "total",
        "title": "Monthly LP: Haifa as a Whole",
    },
    "Ashdod_Legacy_Q": {
        "label": "Ashdod-Legacy",
        "group": "Ashdod",
        "kind": "legacy",
        "title": "Monthly LP: Ashdod-Legacy",
    },
    "Ashdod_HCT_Q": {
        "label": "Ashdod-HCT",
        "group": "Ashdod",
        "kind": "entrant",
        "title": "Monthly LP: Ashdod Entrant (HCT)",
    },
    "Ashdod_port_M": {
        "label": "Ashdod total",
        "group": "Ashdod",
        "kind": "total",
        "title": "Monthly LP: Ashdod as a Whole",
    },
}

PLOT_ORDER = [
    "Haifa_Legacy_Q",
    "Haifa_SIPG_Q",
    "Haifa_port_M",
    "Ashdod_Legacy_Q",
    "Ashdod_HCT_Q",
    "Ashdod_port_M",
]

LINESTYLE = {
    "Haifa_Legacy_Q": "--",
    "Haifa_SIPG_Q": "--",
    "Haifa_port_M": "-",
    "Ashdod_Legacy_Q": "--",
    "Ashdod_HCT_Q": "--",
    "Ashdod_port_M": "-",
}

MARKER = {
    "Haifa_Legacy_Q": "o",
    "Haifa_SIPG_Q": "s",
    "Haifa_port_M": "",
    "Ashdod_Legacy_Q": "o",
    "Ashdod_HCT_Q": "s",
    "Ashdod_port_M": "",
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


def read_tsv(path):
    return pd.read_csv(path, sep="\t")


def find_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find any of these files:\n" + "\n".join(str(p) for p in paths)
    )


def q_to_months(qstr):
    q = str(qstr).strip().upper()
    mapping = {
        "Q1": [1, 2, 3],
        "Q2": [4, 5, 6],
        "Q3": [7, 8, 9],
        "Q4": [10, 11, 12],
    }
    return mapping[q]


def expand_quarterly_to_monthly(df):
    df = df.copy()
    monthly_rows = []
    for _, row in df.iterrows():
        if str(row.get("freq", "")).upper() == "Q":
            months = q_to_months(row["quarter"])
            for month in months:
                new_row = row.copy()
                new_row["freq"] = "M"
                new_row["month"] = month
                new_row["month_index"] = int(row["year"]) * 12 + int(month)
                monthly_rows.append(new_row)
        else:
            monthly_rows.append(row.copy())

    out = pd.DataFrame(monthly_rows)
    return out


def month_end_timestamp(year, month):
    last_day = calendar.monthrange(int(year), int(month))[1]
    return pd.Timestamp(int(year), int(month), last_day)


def load_monthly_panel(panel_path):
    df = read_tsv(panel_path).copy()

    required = {"series_id", "port", "year", "LP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"LP panel missing columns: {sorted(missing)}")

    if "freq" in df.columns and (df["freq"].astype(str).str.upper() == "Q").any():
        df = expand_quarterly_to_monthly(df)

    if "month" not in df.columns:
        raise ValueError("Monthly plotting needs a month column after expansion.")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df["LP"] = pd.to_numeric(df["LP"], errors="coerce")

    df = df.dropna(subset=["year", "month", "LP"]).copy()

    df["month_str"] = (
        df["year"].astype(int).astype(str)
        + "-"
        + df["month"].astype(int).astype(str).str.zfill(2)
    )

    df = df[(df["month_str"] >= SAMPLE_START) & (df["month_str"] <= SAMPLE_END)].copy()
    df["date"] = [month_end_timestamp(y, m) for y, m in zip(df["year"], df["month"])]
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)

    return df


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def add_event_lines(ax, events):
    if not events:
        return

    ymin, ymax = ax.get_ylim()
    text_y = ymax * 0.97

    for event in events:
        dt = event["date"]
        label = event["label"]

        ax.axvline(
            dt,
            linestyle="--",
            linewidth=1.0,
            color="k",
            alpha=0.7,
        )

        ax.text(
            dt,
            text_y,
            label,
            rotation=90,
            va="top",
            ha="right",
            fontsize=11,
        )


def configure_date_axis(ax, df):
    dmin = pd.to_datetime(df["date"].min()).normalize()
    dmax = pd.to_datetime(df["date"].max()).normalize()

    start_year = dmin.year
    end_year = dmax.year

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

    ax.tick_params(axis="x", which="major", length=6)
    ax.tick_params(axis="x", which="minor", length=3, labelsize=9, pad=2)

    ax.grid(True, which="major", axis="x", alpha=0.15)
    ax.grid(True, which="minor", axis="x", alpha=0.08)

    ax.set_xlim(pd.Timestamp(start_year, 1, 1), pd.Timestamp(end_year + 1, 1, 1))


def prep_plot(df):
    return df.copy().sort_values("date").reset_index(drop=True)


def plot_single(df, series_id, outpath, events):
    sub = prep_plot(df.loc[df["series_id"] == series_id].copy())
    if sub.empty:
        return None

    meta = SERIES_META[series_id]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        sub["date"],
        sub["LP"],
        linestyle=LINESTYLE.get(series_id, "-"),
        marker=MARKER.get(series_id, ""),
        linewidth=2,
        label=meta["label"],
    )

    ax.set_title(meta["title"], loc="left", fontsize=13, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Labor productivity (LP)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    configure_date_axis(ax, sub)
    add_event_lines(ax, events)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_multi(df, series_ids, title, outpath, events):
    fig, ax = plt.subplots(figsize=(12, 6))

    plotted_any = False
    combined = []

    for series_id in series_ids:
        sub = prep_plot(df.loc[df["series_id"] == series_id].copy())
        if sub.empty:
            continue

        plotted_any = True
        combined.append(sub)

        ax.plot(
            sub["date"],
            sub["LP"],
            linestyle=LINESTYLE.get(series_id, "-"),
            marker=MARKER.get(series_id, ""),
            linewidth=2,
            label=SERIES_META[series_id]["label"],
        )

    if not plotted_any:
        plt.close(fig)
        return None

    big = pd.concat(combined, ignore_index=True)

    ax.set_title(title, loc="left", fontsize=13, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Labor productivity (LP)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    configure_date_axis(ax, big)
    add_event_lines(ax, events)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return outpath


def main():
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parents[2]

    panel_candidates = [
        root / "Data" / "LP" / "LP_Panel_monthly.tsv",
        root / "Data" / "LP" / "LP_Panel.tsv",
    ]

    panel_path = find_first_existing(panel_candidates)
    out_dir = root / "Design" / "Output (new)" / "LP" / "Visuals"
    ensure_dir(out_dir)

    df = load_monthly_panel(panel_path)

    outputs = []

    outputs.append(plot_single(
        df,
        "Haifa_Legacy_Q",
        out_dir / "plot_lp_haifa_legacy.png",
        REFORM_EVENTS["Haifa"],
    ))

    outputs.append(plot_single(
        df,
        "Haifa_SIPG_Q",
        out_dir / "plot_lp_haifa_entrant.png",
        REFORM_EVENTS["Haifa"],
    ))

    outputs.append(plot_single(
        df,
        "Haifa_port_M",
        out_dir / "plot_lp_haifa_total.png",
        REFORM_EVENTS["Haifa"],
    ))

    outputs.append(plot_multi(
        df,
        ["Haifa_Legacy_Q", "Haifa_SIPG_Q", "Haifa_port_M"],
        "Monthly LP: All Haifa Series",
        out_dir / "plot_lp_haifa_all.png",
        REFORM_EVENTS["Haifa"],
    ))

    outputs.append(plot_single(
        df,
        "Ashdod_Legacy_Q",
        out_dir / "plot_lp_ashdod_legacy.png",
        REFORM_EVENTS["Ashdod"],
    ))

    outputs.append(plot_single(
        df,
        "Ashdod_HCT_Q",
        out_dir / "plot_lp_ashdod_entrant.png",
        REFORM_EVENTS["Ashdod"],
    ))

    outputs.append(plot_single(
        df,
        "Ashdod_port_M",
        out_dir / "plot_lp_ashdod_total.png",
        REFORM_EVENTS["Ashdod"],
    ))

    outputs.append(plot_multi(
        df,
        ["Ashdod_Legacy_Q", "Ashdod_HCT_Q", "Ashdod_port_M"],
        "Monthly LP: All Ashdod Series",
        out_dir / "plot_lp_ashdod_all.png",
        REFORM_EVENTS["Ashdod"],
    ))

    outputs.append(plot_multi(
        df,
        PLOT_ORDER,
        "Monthly LP: All Main Series",
        out_dir / "plot_lp_all_series.png",
        REFORM_EVENTS["All"],
    ))

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
