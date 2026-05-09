#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot_L_Proxy_By_Strategy.py
---------------------------
Strategy-aware visualizer for labor proxy outputs.

Main idea:
- You pass a strategy name such as "common_rule"
- The script looks inside Data/L_proxy/<strategy>/
- It finds that strategy's L_Proxy file
- It writes the same style of visuals and tidy TSVs into:
      Data/L_proxy/<strategy>/Visualizations/

Default behavior for the current implementation:
    --strategy common_rule
reads:
    Data/L_proxy/common_rule/L_Proxy_commonrule_v1.tsv
writes:
    Data/L_proxy/common_rule/Visualizations/

Usage examples:
    python Data/L_proxy/Plot_L_Proxy_By_Strategy.py --strategy common_rule

    python Data/L_proxy/Plot_L_Proxy_By_Strategy.py \
      --strategy common_rule \
      --lproxy Data/L_proxy/common_rule/L_Proxy_commonrule_v1.tsv \
      --outdir Data/L_proxy/common_rule/Visualizations
"""

from __future__ import annotations

import argparse
import calendar
import json
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
    here = Path(__file__).resolve()
    for candidate in [here.parent] + list(here.parents):
        if (candidate / "Data").exists() and (candidate / "Design").exists():
            return candidate
    return Path.cwd().resolve()


def resolve_path(path_str: str | None, repo_root: Path) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def month_end_timestamp(year: int, month: int) -> pd.Timestamp:
    last_day = calendar.monthrange(int(year), int(month))[1]
    return pd.Timestamp(year=int(year), month=int(month), day=last_day)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="common_rule_v6_tonsonly")
    ap.add_argument("--lproxy", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--start", default="2018-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--width", type=float, default=9.5)
    ap.add_argument("--height", type=float, default=5.5)
    ap.add_argument("--ymax", type=float, default=None)
    return ap.parse_args()


def strategy_filename_candidates(strategy: str) -> list[str]:
    compact = strategy.replace("_", "").replace("-", "")
    return [
        f"L_Proxy_{strategy}_v6_tonsonly.tsv",
        f"L_Proxy_{strategy}.tsv",
        f"L_Proxy_{compact}_v6_tonsonly.tsv",
        f"L_Proxy_{compact}.tsv",
        "L_Proxy.tsv",
    ]


def resolve_strategy_lproxy(repo_root: Path, strategy: str, explicit_lproxy: Path | None) -> Path:
    if explicit_lproxy is not None:
        return explicit_lproxy

    strategy_dir = repo_root / "Data" / "L_proxy" / strategy
    candidates = [strategy_dir / name for name in strategy_filename_candidates(strategy)]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find an L_Proxy file for this strategy.\n"
        f"Strategy folder: {strategy_dir}\n"
        "Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def resolve_strategy_outdir(repo_root: Path, strategy: str, explicit_outdir: Path | None) -> Path:
    if explicit_outdir is not None:
        return explicit_outdir
    return repo_root / "Data" / "L_proxy" / strategy / "Visualizations"


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
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "month", "L_hours_i_m"]).copy()

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["L_hours_i_m"] = pd.to_numeric(df["L_hours_i_m"], errors="coerce")
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


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def style_axis(ax: plt.Axes, ylabel: str, ymax: float | None) -> None:
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
    style_axis(ax, "Monthly labor hours (L)", ymax)
    ax.legend(frameon=False, ncols=1)


def plot_port_subset(ax: plt.Axes, port_df: pd.DataFrame, ports: list[str], ymax: float | None) -> None:
    for port in ports:
        sub = port_df[port_df["port"] == port]
        if sub.empty:
            continue
        ax.plot(
            sub["date"],
            sub["L_hours_port"],
            linewidth=2.2,
            label=port,
        )
    style_axis(ax, "Monthly labor hours (L)", ymax)
    ax.legend(frameon=False, ncols=1)


def main() -> None:
    args = parse_args()
    repo_root = infer_repo_root()

    lproxy_path = resolve_strategy_lproxy(
        repo_root=repo_root,
        strategy=args.strategy,
        explicit_lproxy=resolve_path(args.lproxy, repo_root),
    )
    outdir = resolve_strategy_outdir(
        repo_root=repo_root,
        strategy=args.strategy,
        explicit_outdir=resolve_path(args.outdir, repo_root),
    )
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_lproxy(lproxy_path, args.start, args.end)
    term_df = build_terminal_month_panel(raw)
    port_df = build_port_month_panel(term_df)

    write_tsv(term_df, outdir / "L_proxy_monthly_terminals_tidy.tsv")
    write_tsv(port_df, outdir / "L_proxy_monthly_ports_tidy.tsv")

    # Haifa terminals
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    plot_terminal_subset(ax, term_df, ["Haifa-Legacy", "Haifa-Bayport"], args.ymax)
    ax.set_title("Haifa - monthly labor-hours proxy", loc="left", fontsize=12, weight="bold")
    configure_month_ticks(ax, term_df)
    add_reform_lines(ax, REFORM_EVENTS["Haifa"])
    fig.tight_layout()
    fig.savefig(outdir / "haifa_l_proxy_terminals.png", dpi=args.dpi)
    plt.close(fig)

    # Ashdod terminals
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    plot_terminal_subset(ax, term_df, ["Ashdod-Legacy", "Ashdod-HCT"], args.ymax)
    ax.set_title("Ashdod - monthly labor-hours proxy", loc="left", fontsize=12, weight="bold")
    configure_month_ticks(ax, term_df)
    add_reform_lines(ax, REFORM_EVENTS["Ashdod"])
    fig.tight_layout()
    fig.savefig(outdir / "ashdod_l_proxy_terminals.png", dpi=args.dpi)
    plt.close(fig)

    # Port totals
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    plot_port_subset(ax, port_df, PORT_ORDER, args.ymax)
    ax.set_title("Ports - monthly labor-hours proxy", loc="left", fontsize=12, weight="bold")
    configure_month_ticks(ax, port_df)
    add_reform_lines(ax, REFORM_EVENTS["All"])
    fig.tight_layout()
    fig.savefig(outdir / "port_monthly_l_proxy.png", dpi=args.dpi)
    plt.close(fig)

    # All terminals
    fig, ax = plt.subplots(figsize=(args.width + 1.5, args.height + 0.5))
    plot_terminal_subset(ax, term_df, TERMINAL_ORDER, args.ymax)
    ax.set_title("All terminals - monthly labor-hours proxy", loc="left", fontsize=12, weight="bold")
    configure_month_ticks(ax, term_df)
    add_reform_lines(ax, REFORM_EVENTS["All"])
    fig.tight_layout()
    fig.savefig(outdir / "all_terminal_monthly_l_proxy.png", dpi=args.dpi)
    plt.close(fig)

    manifest = {
        "strategy": args.strategy,
        "lproxy_path": str(lproxy_path),
        "outdir": str(outdir),
        "start": args.start,
        "end": args.end,
        "outputs": [
            str(outdir / "haifa_l_proxy_terminals.png"),
            str(outdir / "ashdod_l_proxy_terminals.png"),
            str(outdir / "port_monthly_l_proxy.png"),
            str(outdir / "all_terminal_monthly_l_proxy.png"),
            str(outdir / "L_proxy_monthly_terminals_tidy.tsv"),
            str(outdir / "L_proxy_monthly_ports_tidy.tsv"),
        ],
    }
    (outdir / "_viz_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[viz] strategy: {args.strategy}")
    print(f"[viz] read:     {lproxy_path}")
    print(f"[viz] wrote:    {outdir}")


if __name__ == "__main__":
    main()