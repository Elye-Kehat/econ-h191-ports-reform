#!/usr/bin/env python3
"""
Plot_Model_1B_event_study.py

Reads Model 1B pooled dynamic betas TSVs and saves *combined* coefficient-path plots.

Default output: 4 plots total
  - NYT  x {haifa_comp, haifa_priv}
  - TWFE x {haifa_comp, haifa_priv}

Each plot overlays all relevant targets for that reform (within the chosen spec(s)).

Inputs (expected):
  Thesis/Design/Output (new)/Model_1B/model1b_kl_dynamic_betas_all.tsv
  Thesis/Design/Output (new)/Model_1B/model1b_kl_dynamic_betas_all_twfe.tsv

Outputs:
  Thesis/Design/Output (new)/Model_1B/Figures/combined/
    model1b_kl_eventstudy_nyt_haifa_comp_porttr.png
    model1b_kl_eventstudy_nyt_haifa_priv_porttr.png
    model1b_kl_eventstudy_twfe_haifa_comp_porttr.png
    model1b_kl_eventstudy_twfe_haifa_priv_porttr.png

Usage:
  python Plot_Model_1B_event_study.py
  python Plot_Model_1B_event_study.py --spec baseline
  python Plot_Model_1B_event_study.py --spec all
  python Plot_Model_1B_event_study.py --jmin -24 --rotate 45
  python Plot_Model_1B_event_study.py --jmax 24   (cap to 2-year horizon)
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

    # v2 Model_1B writes: reform, target, spec_name, event_time, j, beta_hat, se, ...
    need = {"reform", "target", "spec_name", "event_time", "beta_hat", "se"}
    missing = need.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")

    if "j" not in df.columns:
        df["j"] = df["event_time"]

    df["j"] = pd.to_numeric(df["j"], errors="coerce").astype(int)
    df["beta_hat"] = pd.to_numeric(df["beta_hat"], errors="coerce")
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

    # Clip range for plotting only
    d = d[(d["j"] >= jmin) & (d["j"] <= jmax)].copy()
    if d.empty:
        print(f"[SKIP] All rows clipped out for design={design}, reform={reform} (jmin={jmin}, jmax={jmax})")
        return

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.axhline(0.0, linewidth=1)
    ax.axvline(0.0, linewidth=1)

    # One line per (target, spec_name)
    for (target, spec_name), g in d.groupby(["target", "spec_name"]):
        g = g.sort_values("j")
        j = g["j"].to_numpy()
        beta = g["beta_hat"].to_numpy()
        se = g["se"].to_numpy()
        yerr = 1.96 * se

        label = f"{target} — {spec_name}"
        ax.errorbar(
            j, beta, yerr=yerr,
            fmt="o-", linewidth=1, markersize=4, capsize=2,
            label=label
        )

    step = choose_tick_step(jmin, jmax)
    ticks = list(range(jmin, jmax + 1, step))
    ax.set_xticks(ticks)
    if rotate != 0:
        for lab in ax.get_xticklabels():
            lab.set_rotation(rotate)
            lab.set_ha("right")

    title_specs = ", ".join(specs_to_plot)
    ax.set_title(f"Model 1B ({design.upper()}): {reform} — dynamic effects on ln(K/L)  |  specs: {title_specs}")
    ax.set_xlabel("Event time j (months relative to event; j=0 is event month; j=-1 omitted)")
    ax.set_ylabel("Coefficient (log points)")
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
                    help="Minimum j shown (default -12). Use -24 for full pre-period debugging.")
    ap.add_argument("--jmax", type=int, default=None,
                    help="Maximum j shown (default None = use max available in each file).")
    ap.add_argument("--rotate", type=int, default=0,
                    help="Rotate x tick labels (degrees). Useful if you choose jmin=-24.")
    args = ap.parse_args()

    thesis_root = find_thesis_root(Path(__file__).resolve())
    model_dir = thesis_root / "Design" / "Output (new)" / "Model_1B"
    fig_dir = model_dir / "Figures" / "combined"
    fig_dir.mkdir(parents=True, exist_ok=True)

    nyt_path = model_dir / "model1b_kl_dynamic_betas_all.tsv"
    twfe_path = model_dir / "model1b_kl_dynamic_betas_all_twfe.tsv"

    if not nyt_path.exists():
        raise FileNotFoundError(f"Missing NYT dynamic betas: {nyt_path}")
    if not twfe_path.exists():
        raise FileNotFoundError(f"Missing TWFE dynamic betas: {twfe_path}")

    df_nyt = load_dynamic(nyt_path)
    df_twfe = load_dynamic(twfe_path)

    specs = ["baseline", "porttr", "tr_shocks"] if args.spec == "all" else [args.spec]

    # Plot only reforms present in the files (keeps future expansion clean)
    reforms_nyt = sorted(df_nyt["reform"].unique().tolist())
    reforms_twfe = sorted(df_twfe["reform"].unique().tolist())
    reforms = sorted(set(reforms_nyt).intersection(set(reforms_twfe)))
    if not reforms:
        reforms = sorted(set(reforms_nyt + reforms_twfe))

    for reform in reforms:
        out = fig_dir / sanitize_filename(f"model1b_kl_eventstudy_nyt_{reform}_{args.spec}.png")
        make_plot(df_nyt, "nyt", reform, specs, out, args.jmin, args.jmax, args.rotate)

    for reform in reforms:
        out = fig_dir / sanitize_filename(f"model1b_kl_eventstudy_twfe_{reform}_{args.spec}.png")
        make_plot(df_twfe, "twfe", reform, specs, out, args.jmin, args.jmax, args.rotate)


if __name__ == "__main__":
    main()