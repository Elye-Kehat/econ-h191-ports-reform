#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot_L_Proxy_Series.py
----------------------
Make publication-ready monthly plots for the labor-hours proxy built in L_Proxy.tsv.

Default outputs:
  Design/Output (new)/Visualizations/haifa_l_proxy_terminals.png
  Design/Output (new)/Visualizations/ashdod_l_proxy_terminals.png
  Design/Output (new)/Visualizations/port_monthly_l_proxy.png
  Design/Output (new)/Visualizations/all_terminal_monthly_l_proxy.png
  Design/Output (new)/Visualizations/L_proxy_monthly_terminals_tidy.tsv
  Design/Output (new)/Visualizations/L_proxy_monthly_ports_tidy.tsv

Suggested location in the repo:
  Design/Code/Visualization/Plot_L_Proxy_Series.py

Usage example:
  python "Design/Code/Visualization/Plot_L_Proxy_Series.py" \
    --lproxy "Data/L_proxy/L_Proxy.tsv" \
    --outdir "Design/Output (new)/Visualizations" \
    --dpi 220
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


TERMINAL_ORDER = [
    "Haifa-Legacy",
    "Haifa-Bayport",
    "Ashdod-Legacy",
    "Ashdod-HCT",
]

TERMINAL_LABELS = {
    "Haifa-Legacy": "Haifa-Legacy",
    "Haifa-Bayport": "Haifa-Bayport",
    "Ashdod-Legacy": "Ashdod-Legacy",
    "Ashdod-HCT": "Ashdod-HCT",
}

TERMINAL_LINESTYLE = {
    "Haifa-Legacy": "-",
    "Haifa-Bayport": "--",
    "Ashdod-Legacy": "-",
    "Ashdod-HCT": "--",
}

PORT_ORDER = ["Haifa", "Ashdod"]

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


def infer_repo_root() -> Path:
    """
    If the script is placed at THESIS/Design/Code/Visualization/Plot_L_Proxy_Series.py,
    then parents[3] is THESIS/. Fall back to cwd if not enough levels exist.
    """
    here = Path(__file__).resolve()
    try:
        return here.parents[3]
    except IndexError:
        return Path.cwd().resolve()


def resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (repo_root / path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lproxy", default="Data/L_proxy/L_Proxy.tsv")
    ap.add_argument("--outdir", default="Design/Output (new)/Visualizations")
    ap.add_argument("--start", default="2018-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--width", type=float, default=9.5)
    ap.add_argument("--height", type=float, default=5.5)
    ap.add_argument("--ymax", type=float, default=None, help="Optional common y-axis max")
    return ap.parse_args()


def month_end_timestamp(year: int, month: int) -> pd.Timestamp:
    last_day = calendar.monthrange(int(year), int(month))[1]
    return pd.Timestamp(year=int(year), month=int(month), day=last_day)


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


def configure_month_ticks(ax: plt.Axes, df: pd.DataFrame) -> None:
    dmin = pd.to_datetime(df["date"].min()).normalize()
    dmax = pd.to_datetime(df["date"].max()).normalize()
    start_year = dmin.year
    end_year = dmax.year

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

    ax.tick_params(axis="x", which="major", length=6)
    ax.tick_params(axis="x", which="minor", length=3, labelsize=8, pad=2)

    ax.grid(True, which="major", axis="x", alpha=0.15)
    ax.grid(True, which="minor", axis="x", alpha=0.08)

    ax.set_xlim(pd.Timestamp(start_year, 1, 1), pd.Timestamp(end_year + 1, 1, 1))


def load_lproxy(lproxy_path: Path, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(lproxy_path, sep="\t")

    required = {"port", "terminal", "year", "month", "L_hours_i_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"L_Proxy missing columns: {sorted(missing)}")

    df = df.copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["date"] = df.apply(lambda r: month_end_timestamp(r["year"], r["month"]), axis=1)

    start_ts = pd.Timestamp(f"{start}-01")
    end_ts = pd.Timestamp(f"{end}-01") + pd.offsets.MonthEnd(0)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()

    df = df.sort_values(["port", "terminal", "date"]).reset_index(drop=True)
    return df


def build_terminal_month_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["port", "terminal", "year", "month", "date"], as_index=False)["L_hours_i_m"]
          .sum()
          .rename(columns={"L_hours_i_m": "L_hours"})
    )
    return out.sort_values(["port", "terminal", "date"]).reset_index(drop=True)


def build_port_month_panel(term_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        term_df.groupby(["port", "year", "month", "date"], as_index=False)["L_hours"]
               .sum()
               .rename(columns={"L_hours": "L_hours_port"})
    )
    return out.sort_values(["port", "date"]).reset_index(drop=True)


def style_axis(ax: plt.Axes, title: str, ylabel: str, ymax: float | None) -> None:
    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if ymax is not None:
        ax.set_ylim(0, ymax)


def plot_terminal_subset(ax: plt.Axes, term_df: pd.DataFrame, terminals: list[str], ymax: float | None) -> None:
    for terminal in terminals:
        sub = term_df[term_df["terminal"] == terminal]
        if sub.empty:
            continue
        ax.plot(
            sub["date"],
            sub["L_hours"],
            linewidth=2.0,
            linestyle=TERMINAL_LINESTYLE.get(terminal, "-"),
            label=TERMINAL_LABELS.get(terminal, terminal),
        )
    style_axis(ax, "", "Monthly labor hours (L)", ymax)
    ax.legend(frameon=False, ncols=1)


def plot_port_subset(ax: plt.Axes, port_df: pd.DataFrame, ports: list[str], ymax: float | None) -> None:
    for port in ports:
        sub = port_df[port_df["port"] == port]
        if sub.empty:
            continue
        ax.plot(sub["date"], sub["L_hours_port"], linewidth=2.2, label=port)
    style_axis(ax, "", "Monthly labor hours (L)", ymax)
    ax.legend(frameon=False, ncols=1)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def main() -> None:
    args = parse_args()
    repo_root = infer_repo_root()
    lproxy_path = resolve_path(args.lproxy, repo_root)
    outdir = resolve_path(args.outdir, repo_root)
    outdir.mkdir(parents=True, exist_ok=True)

    term_df = build_terminal_month_panel(load_lproxy(lproxy_path, args.start, args.end))
    port_df = build_port_month_panel(term_df)

    write_tsv(term_df, outdir / "L_proxy_monthly_terminals_tidy.tsv")
    write_tsv(port_df, outdir / "L_proxy_monthly_ports_tidy.tsv")

    # Haifa terminals
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    plot_terminal_subset(ax, term_df, ["Haifa-Legacy", "Haifa-Bayport"], args.ymax)
    ax.set_title("Haifa — monthly labor-hours proxy", loc="left", fontsize=12, weight="bold")
    configure_month_ticks(ax, term_df)
    add_reform_lines(ax, REFORM_EVENTS["Haifa"])
    fig.tight_layout()
    fig.savefig(outdir / "haifa_l_proxy_terminals.png", dpi=args.dpi)
    plt.close(fig)

    # Ashdod terminals
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    plot_terminal_subset(ax, term_df, ["Ashdod-Legacy", "Ashdod-HCT"], args.ymax)
    ax.set_title("Ashdod — monthly labor-hours proxy", loc="left", fontsize=12, weight="bold")
    configure_month_ticks(ax, term_df)
    add_reform_lines(ax, REFORM_EVENTS["Ashdod"])
    fig.tight_layout()
    fig.savefig(outdir / "ashdod_l_proxy_terminals.png", dpi=args.dpi)
    plt.close(fig)

    # Port totals
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    plot_port_subset(ax, port_df, PORT_ORDER, args.ymax)
    ax.set_title("Ports — monthly labor-hours proxy", loc="left", fontsize=12, weight="bold")
    configure_month_ticks(ax, port_df)
    add_reform_lines(ax, REFORM_EVENTS["All"])
    fig.tight_layout()
    fig.savefig(outdir / "port_monthly_l_proxy.png", dpi=args.dpi)
    plt.close(fig)

    # All terminal series
    fig, ax = plt.subplots(figsize=(args.width + 1.5, args.height + 0.5))
    plot_terminal_subset(ax, term_df, TERMINAL_ORDER, args.ymax)
    ax.set_title("All terminals — monthly labor-hours proxy", loc="left", fontsize=12, weight="bold")
    configure_month_ticks(ax, term_df)
    add_reform_lines(ax, REFORM_EVENTS["All"])
    fig.tight_layout()
    fig.savefig(outdir / "all_terminal_monthly_l_proxy.png", dpi=args.dpi)
    plt.close(fig)

    print(f"[viz] wrote L-proxy plots and tidy exports to: {outdir}")


if __name__ == "__main__":
    main()
