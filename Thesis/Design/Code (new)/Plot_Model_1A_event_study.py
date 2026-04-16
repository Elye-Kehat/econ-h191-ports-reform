#!/usr/bin/env python3
"""
Plot_Model_1A_event_study.py  (v2)

Goal: produce a small number of readable event-study plots for Model 1A.

Outputs (default): exactly 4 plots total
  - NYT  x {haifa_comp, haifa_priv}
  - TWFE x {haifa_comp, haifa_priv}

Each plot combines all "regressions" (targets) for the given reform clock:
  - haifa_comp: Haifa-Legacy terminal + Haifa-Bayport terminal in the same plot
  - haifa_priv: Haifa-Legacy terminal (only)

Formatting improvements:
  - clip plotted j-range (default jmin=-12, jmax=None -> uses max available in data)
  - fewer x-ticks + optional rotation for readability
  - error bars (95% CI) rather than huge CI bands for multiple series overlays

Usage:
  python Plot_Model_1A_event_study.py
  python Plot_Model_1A_event_study.py --spec porttr
  python Plot_Model_1A_event_study.py --spec all
  python Plot_Model_1A_event_study.py --jmin -24 --rotate 45
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_thesis_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "Data").exists() and (p / "Design").exists():
            return p
    raise RuntimeError("Could not locate Thesis root (expected directories: Data/ and Design/).")


def sanitize_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s


def load_dynamic(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    need = {"reform", "target", "spec_name", "event_time", "beta", "se"}
    missing = need.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")
    if "j" not in df.columns:
        df["j"] = df["event_time"]
    df["j"] = pd.to_numeric(df["j"], errors="coerce").astype(int)
    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    df["se"] = pd.to_numeric(df["se"], errors="coerce")
    return df


def choose_tick_step(jmin: int, jmax: int) -> int:
    span = jmax - jmin
    if span <= 18:
        return 1
    if span <= 30:
        return 2
    if span <= 48:
        return 3
    return 4


def make_plot(
    df: pd.DataFrame,
    design: str,
    reform: str,
    specs_to_plot: list[str],
    out_path: Path,
    jmin: int,
    jmax_override: int | None,
    rotate: int,
) -> None:
    d = df[(df["reform"] == reform) & (df["spec_name"].isin(specs_to_plot))].copy()
    if d.empty:
        print(f"[SKIP] No rows for design={design}, reform={reform}, specs={specs_to_plot}")
        return

    # Determine max j to plot
    jmax_data = int(d["j"].max())
    jmax = int(jmax_override) if jmax_override is not None else jmax_data

    # Clip range for plotting (this is just visualization; regressions unchanged)
    d = d[(d["j"] >= jmin) & (d["j"] <= jmax)].copy()
    if d.empty:
        print(f"[SKIP] All rows clipped out for design={design}, reform={reform} (jmin={jmin}, jmax={jmax})")
        return

    # Plot
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.axhline(0.0, linewidth=1)
    ax.axvline(0.0, linewidth=1)

    # One line per (target, spec_name)
    for (target, spec_name), g in d.groupby(["target", "spec_name"]):
        g = g.sort_values("j")
        j = g["j"].to_numpy()
        beta = g["beta"].to_numpy()
        se = g["se"].to_numpy()

        # 95% CI error bars; keep it readable by not adding caps that are too thick
        yerr = 1.96 * se
        label = f"{target} — {spec_name}"
        ax.errorbar(j, beta, yerr=yerr, fmt="o-", linewidth=1, markersize=4, capsize=2, label=label)

    # Ticks (avoid the unreadable “every integer tick label” pile-up)
    step = choose_tick_step(jmin, jmax)
    ticks = list(range(jmin, jmax + 1, step))
    ax.set_xticks(ticks)
    if rotate != 0:
        for lab in ax.get_xticklabels():
            lab.set_rotation(rotate)
            lab.set_ha("right")

    ax.set_xlabel("Event time j (months relative to event; j=0 is event month; j=-1 omitted)")
    ax.set_ylabel("Coefficient (log points)")
    title_specs = ", ".join(specs_to_plot)
    ax.set_title(f"Model 1A ({design.upper()}): {reform} — dynamic effects on ln(LP)  |  specs: {title_specs}")

    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="porttr", choices=["baseline", "porttr", "tr_shocks", "all"],
                    help="Which spec(s) to plot. Default 'porttr' (preferred).")
    ap.add_argument("--jmin", type=int, default=-12,
                    help="Minimum j to show on x-axis (default -12). Use -24 to reproduce full pre-period.")
    ap.add_argument("--jmax", type=int, default=None,
                    help="Maximum j to show on x-axis. Default None = use max available in data for that design+reform.")
    ap.add_argument("--rotate", type=int, default=0,
                    help="Rotate x tick labels (degrees). Useful if you choose jmin=-24.")
    args = ap.parse_args()

    thesis_root = find_thesis_root(Path(__file__).resolve())
    model_dir = thesis_root / "Design" / "Output (new)" / "Model_1A"
    fig_dir = model_dir / "Figures" / "combined"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Inputs
    nyt_path = model_dir / "model1a_lp_dynamic_betas_all.tsv"
    twfe_path = model_dir / "model1a_lp_dynamic_betas_all_twfe.tsv"

    if not nyt_path.exists():
        raise FileNotFoundError(f"Missing NYT dynamic betas: {nyt_path}")
    if not twfe_path.exists():
        raise FileNotFoundError(f"Missing TWFE dynamic betas: {twfe_path}")

    df_nyt = load_dynamic(nyt_path)
    df_twfe = load_dynamic(twfe_path)

    if args.spec == "all":
        specs = ["baseline", "porttr", "tr_shocks"]
    else:
        specs = [args.spec]

    reforms = ["haifa_comp", "haifa_priv"]  # your current Model 1A(v3) universe

    # NYT plots (2)
    for reform in reforms:
        out = fig_dir / sanitize_filename(f"model1a_lp_eventstudy_nyt_{reform}_{args.spec}.png")
        make_plot(df_nyt, "nyt", reform, specs, out, args.jmin, args.jmax, args.rotate)

    # TWFE plots (2)
    for reform in reforms:
        out = fig_dir / sanitize_filename(f"model1a_lp_eventstudy_twfe_{reform}_{args.spec}.png")
        make_plot(df_twfe, "twfe", reform, specs, out, args.jmin, args.jmax, args.rotate)


if __name__ == "__main__":
    main()