#!/usr/bin/env python3
"""
Plot_KL_Series_v5_clipped_reform_lines_quarter_ticks.py

Creates a thesis-ready raw K/L figure for the main Haifa series with:
- capped y-axis
- clipped outlier annotation moved to the RIGHT of the spike
- Haifa reform lines
- quarterly x-axis ticks
- fixed plotting window from 2018Q1 through 2024Q4
"""

from __future__ import annotations

from matplotlib.ticker import FuncFormatter
import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT_CANDIDATES = [
    Path("Data/KL/common_rule_v6_tonsonly/KL_Panel_monthly.tsv"),
    Path("Data/KL/KL_Panel_monthly.tsv"),
    Path("KL_Panel_monthly.tsv"),
]

DEFAULT_OUTPUT_DIR_CANDIDATES = [
    Path("Data/KL/common_rule_v6_tonsonly"),
    Path("Data/KL"),
    Path("."),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="", help="Path to KL_Panel_monthly.tsv")
    p.add_argument("--out-dir", default="", help="Directory for output figures")
    p.add_argument("--ymax", type=float, default=10000.0, help="Upper y limit for capped plot")
    p.add_argument("--start", default="2018-01-01", help="Optional start month/date filter")
    p.add_argument("--end", default="2024-12-31", help="Optional end month/date filter")
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    p.add_argument("--comp-date", default="2021-09-01", help="Haifa competition reform date")
    p.add_argument("--priv-date", default="2023-01-01", help="Haifa privatization reform date")
    p.add_argument("--no-lines", action="store_true", help="Disable reform lines")
    return p.parse_args()


def resolve_input(path_str: str) -> Path:
    if path_str:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(p) for p in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(
        "Could not find a K/L panel automatically. Searched:\n"
        f"{searched}\n"
        "Pass --input explicitly."
    )


def resolve_output_dir(path_str: str) -> Path:
    if path_str:
        out_dir = Path(path_str)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    for candidate in DEFAULT_OUTPUT_DIR_CANDIDATES:
        if candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate

    Path("Data/KL").mkdir(parents=True, exist_ok=True)
    return Path("Data/KL")


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def coerce_date(df: pd.DataFrame) -> pd.Series:
    month_col = find_column(df, ["month", "Month", "date", "Date"])
    year_col = find_column(df, ["year", "Year"])
    month_num_col = find_column(df, ["month_num", "month_number"])

    if month_col is not None:
        s = df[month_col].astype(str).str.strip()
        parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().any():
            return parsed
        parsed = pd.to_datetime(s + "-01", errors="coerce")
        if parsed.notna().any():
            return parsed

    if year_col is not None and month_num_col is not None:
        year = pd.to_numeric(df[year_col], errors="coerce")
        month_num = pd.to_numeric(df[month_num_col], errors="coerce")
        return pd.to_datetime({"year": year, "month": month_num, "day": 1}, errors="coerce")

    if year_col is not None and month_col is not None:
        year = pd.to_numeric(df[year_col], errors="coerce")
        month_num = pd.to_numeric(df[month_col], errors="coerce")
        return pd.to_datetime({"year": year, "month": month_num, "day": 1}, errors="coerce")

    raise ValueError("Could not infer a monthly date column.")


def normalize_series_name(raw: str) -> str:
    s = raw.strip().lower().replace("-", "_").replace(" ", "_")

    if ("legacy" in s or "hpc" in s) and "ashdod" not in s:
        return "Legacy"
    if any(token in s for token in ["bayport", "entrant", "sipg"]) and "ashdod" not in s:
        return "Entrant"
    if any(token in s for token in ["haifa_total", "haifa_port", "haifa_port_kl", "total"]) and "ashdod" not in s:
        return "Haifa total"

    return ""


def load_main_haifa_series(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep="\t")

    series_col = find_column(df, ["series_id", "series", "entity", "name"])
    if series_col is None:
        raise ValueError("Could not find a series identifier column such as series_id or entity.")

    kl_col = find_column(df, ["KL", "kl", "k_l", "K_L", "K_per_L", "K_over_L", "K/L"])
    if kl_col is None:
        raise ValueError("Could not find a K/L column such as KL.")

    out = df.copy()
    out["plot_series"] = out[series_col].astype(str).map(normalize_series_name)
    out = out.loc[out["plot_series"] != ""].copy()

    if out.empty:
        unique_series = ", ".join(sorted(df[series_col].astype(str).unique()))
        raise ValueError(
            "Did not find the main Haifa K/L series in the input panel. "
            f"Available series include: {unique_series}"
        )

    out["date"] = coerce_date(out)
    out["KL_value"] = pd.to_numeric(out[kl_col], errors="coerce")
    out = out.loc[out["date"].notna() & out["KL_value"].notna()].copy()

    order = pd.CategoricalDtype(["Legacy", "Entrant", "Haifa total"], ordered=True)
    out["plot_series"] = out["plot_series"].astype(order)

    return out[["date", "plot_series", "KL_value"]].sort_values(["plot_series", "date"]).reset_index(drop=True)


def apply_date_filter(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    out = df.copy()
    if start:
        out = out.loc[out["date"] >= pd.to_datetime(start)]
    if end:
        out = out.loc[out["date"] <= pd.to_datetime(end)]
    return out.copy()


def format_num(x: float) -> str:
    return f"{x:,.0f}"


def add_reform_lines(ax: plt.Axes, ymax: float, comp_date: str, priv_date: str) -> None:
    comp_dt = pd.to_datetime(comp_date)
    priv_dt = pd.to_datetime(priv_date)

    ax.axvline(comp_dt, linestyle="--", linewidth=1)
    ax.axvline(priv_dt, linestyle="--", linewidth=1)

    ax.text(comp_dt, ymax * 0.97, "Bayport opens (Haifa)", rotation=90, ha="right", va="top", fontsize=10)
    ax.text(priv_dt, ymax * 0.97, "Haifa privatized", rotation=90, ha="right", va="top", fontsize=10)


def quarter_label(x, pos=None):
    dt = mdates.num2date(x)
    month_to_q = {3: 1, 6: 2, 9: 3, 12: 4}
    q = month_to_q.get(dt.month)
    return f"{dt.year}Q{q}" if q is not None else ""


def add_quarter_ticks(ax: plt.Axes) -> None:
    quarter_ticks = pd.date_range("2018-03-01", "2024-12-01", freq="3MS")
    ax.set_xticks(quarter_ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(quarter_label))
    ax.set_xlim(pd.Timestamp("2018-01-01"), pd.Timestamp("2024-12-31"))
    ax.margins(x=0)


def add_clipped_annotations(ax: plt.Axes, df: pd.DataFrame, ymax: float) -> pd.DataFrame:
    clipped = df.loc[df["KL_value"] > ymax].copy()
    if clipped.empty:
        return clipped

    for series_name, g in clipped.groupby("plot_series", observed=True):
        ax.scatter(g["date"], [ymax] * len(g), marker="^", s=42, zorder=5)

        peak = g.loc[g["KL_value"].idxmax()]
        label_x = peak["date"] + pd.DateOffset(months=4)
        label_y = ymax * 0.89

        ax.annotate(
            f"{series_name} peak ≈ {format_num(float(peak['KL_value']))}",
            xy=(peak["date"], ymax),
            xytext=(label_x, label_y),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "lw": 0.8},
        )

    return clipped


def draw_plot(
    df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
    ymax: Optional[float],
    comp_date: str,
    priv_date: str,
    show_lines: bool,
) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(12, 6))

    for series_name, g in df.groupby("plot_series", observed=True):
        y = g["KL_value"] if ymax is None else g["KL_value"].clip(upper=ymax)
        ax.plot(g["date"], y, linewidth=2, label=str(series_name))

    ax.set_title("Monthly K/L: All Main Haifa Series")
    ax.set_xlabel("Month")
    ax.set_ylabel("K/L (thousands of NIS per labor-hour)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    clipped = pd.DataFrame(columns=df.columns)
    upper_for_labels = float(df["KL_value"].max()) if ymax is None else ymax

    if ymax is not None:
        ax.set_ylim(0, ymax)
        clipped = add_clipped_annotations(ax, df, ymax)

    if show_lines:
        add_reform_lines(ax, upper_for_labels, comp_date, priv_date)

    add_quarter_ticks(ax)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return clipped


def main() -> None:
    args = parse_args()

    input_path = resolve_input(args.input)
    out_dir = resolve_output_dir(args.out_dir)

    df = load_main_haifa_series(input_path)
    df = apply_date_filter(df, args.start, args.end)

    if df.empty:
        raise ValueError("No observations remain after filtering.")

    full_png = out_dir / "plot_kl_all_main_full_reform_lines_qticks.png"
    full_pdf = out_dir / "plot_kl_all_main_full_reform_lines_qticks.pdf"
    capped_png = out_dir / "plot_kl_all_main_capped_reform_lines_qticks.png"
    capped_pdf = out_dir / "plot_kl_all_main_capped_reform_lines_qticks.pdf"
    outlier_tsv = out_dir / "plot_kl_all_main_capped_reform_lines_qticks_outliers.tsv"

    draw_plot(
        df=df,
        out_png=full_png,
        out_pdf=full_pdf,
        dpi=args.dpi,
        ymax=None,
        comp_date=args.comp_date,
        priv_date=args.priv_date,
        show_lines=not args.no_lines,
    )

    clipped = draw_plot(
        df=df,
        out_png=capped_png,
        out_pdf=capped_pdf,
        dpi=args.dpi,
        ymax=args.ymax,
        comp_date=args.comp_date,
        priv_date=args.priv_date,
        show_lines=not args.no_lines,
    )

    clipped_export = clipped.copy()
    if not clipped_export.empty:
        clipped_export["date"] = clipped_export["date"].dt.strftime("%Y-%m")
    clipped_export.to_csv(outlier_tsv, sep="\t", index=False)

    print("Done.")
    print(f"Read: {input_path}")
    print(f"Wrote: {full_png}")
    print(f"Wrote: {full_pdf}")
    print(f"Wrote: {capped_png}")
    print(f"Wrote: {capped_pdf}")
    print(f"Wrote: {outlier_tsv}")


if __name__ == "__main__":
    main()