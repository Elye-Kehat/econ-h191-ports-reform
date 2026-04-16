#!/usr/bin/env python
"""Model_1A_to_tables.py

Post-processing script for Model 1A (LP event-study).

Inputs (from Model_1A.py):
  - model1a_lp_dynamic_betas_all.tsv
  - model1a_lp_window_betas_all.tsv
  - model1a_lp_pretrend_tests_all.tsv

Outputs (for LaTeX tables):
  - Tables/model1a_lp_windows_for_tables.tsv
  - Tables/model1a_lp_dynamic_for_appendix.tsv
  - Tables/model1a_lp_pretrend_for_tables.tsv

This script does not re-estimate regressions. It only reshapes and
annotates the aggregated outputs from Model_1A.py, and adds a separate
"post_yr2" window (m in [13,24]) constructed from the dynamic betas.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd


# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
THESIS_ROOT = THIS_FILE.parents[2]

MODEL1A_OUTPUT_DIR = THESIS_ROOT / "Design" / "Output (new)" / "Model_1A"
TABLES_OUTPUT_DIR = MODEL1A_OUTPUT_DIR / "Tables"
TABLES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DYNAMIC_ALL_PATH = MODEL1A_OUTPUT_DIR / "model1a_lp_dynamic_betas_all.tsv"
WINDOWS_ALL_PATH = MODEL1A_OUTPUT_DIR / "model1a_lp_window_betas_all.tsv"
PRETREND_PATH = MODEL1A_OUTPUT_DIR / "model1a_lp_pretrend_tests_all.tsv"


# --------------------------------------------------------------------
# Helpers: normal p-value and significance stars
# --------------------------------------------------------------------


def normal_p_two_sided(t: float) -> float:
    """Two-sided p-value under N(0,1) for a given t-statistic."""
    if not np.isfinite(t):
        return np.nan
    z = abs(float(t))
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return 2.0 * (1.0 - cdf)


def stars_from_p(p: float) -> str:
    """Conventional significance stars based on a p-value."""
    if not np.isfinite(p):
        return ""
    if p <= 0.01:
        return "***"
    if p <= 0.05:
        return "**"
    if p <= 0.10:
        return "*"
    return ""


# --------------------------------------------------------------------
# post_yr2 window constructor (m in [13,24])
# --------------------------------------------------------------------


def add_post_year2_windows(win: pd.DataFrame) -> pd.DataFrame:
    """\
    Add a separate "post_yr2" window (m in [13,24]) to the window-level
    dataframe, computed from the dynamic betas in
    model1a_lp_dynamic_betas_all.tsv.

    If a "post_yr2" row already exists, this is a no-op.
    """
    if "window" in win.columns and (win["window"] == "post_yr2").any():
        return win

    dyn = pd.read_csv(DYNAMIC_ALL_PATH, sep="\t")

    # Harmonize: rename "t" -> "tvalue" if needed (we may not use it here,
    # but this keeps things consistent with other scripts).
    if "t" in dyn.columns and "tvalue" not in dyn.columns:
        dyn = dyn.rename(columns={"t": "tvalue"})

    required_dyn: Set[str] = {
        "reform",
        "target",
        "spec_name",
        "event_time",
        "beta",
        "se",
    }
    missing_dyn = required_dyn.difference(dyn.columns)
    if missing_dyn:
        raise ValueError(
            f"DYNAMIC_ALL_PATH missing columns needed for post_yr2 window: {missing_dyn}"
        )

    group_cols = ["reform", "target", "spec_name"]
    base_groups = win[group_cols].drop_duplicates()

    new_rows = []
    for _, g in base_groups.iterrows():
        ref = g["reform"]
        targ = g["target"]
        spec = g["spec_name"]

        dyn_g = dyn[
            (dyn["reform"] == ref)
            & (dyn["target"] == targ)
            & (dyn["spec_name"] == spec)
        ]

        max_m = dyn_g["event_time"].max()
        upper = 24
        if pd.notnull(max_m) and max_m >= 13:
            upper = int(min(24, max_m))

        dyn_w = dyn_g[(dyn_g["event_time"] >= 13) & (dyn_g["event_time"] <= upper)]



        # Take n_obs and r2 from any existing window row for this spec
        base_rows = win[
            (win["reform"] == ref)
            & (win["target"] == targ)
            & (win["spec_name"] == spec)
        ]
        if base_rows.empty:
            n_obs = np.nan
            r2 = np.nan
        else:
            base_row = base_rows.iloc[0]
            n_obs = base_row["n_obs"]
            r2 = base_row["r2"]

        if dyn_w.empty:
            # No support yet in [13,24] for this spec (e.g. Ashdod, priv)
            beta = np.nan
            se = np.nan
        else:
            k = len(dyn_w)
            beta = dyn_w["beta"].mean()
            # Approximate Var(mean(beta_m)) assuming zero covariance across m
            var_hat = (dyn_w["se"] ** 2).sum() / (k ** 2)
            se = math.sqrt(var_hat)

        new_rows.append(
            {
                "reform": ref,
                "target": targ,
                "spec_name": spec,
                "window": "post_yr2",
                "a": 13,
                "b": 24,
                "beta": beta,
                "se": se,
                "n_obs": n_obs,
                "r2": r2,
            }
        )

    if new_rows:
        win = pd.concat([win, pd.DataFrame(new_rows)], ignore_index=True)

    return win


# --------------------------------------------------------------------
# Main transforms
# --------------------------------------------------------------------


def build_windows_for_tables() -> pd.DataFrame:
    """\
    Take model1a_lp_window_betas_all.tsv and:

      * add a separate "post_yr2" window m∈[13,24],
      * add t-stat, approximate N(0,1) p-value, and significance stars,
      * add simple 95% CIs (normal approximation).

    Output:
      Design/Output (new)/Model_1A/Tables/model1a_lp_windows_for_tables.tsv
    """
    win = pd.read_csv(WINDOWS_ALL_PATH, sep="\t")

    required = {
        "reform",
        "target",
        "spec_name",
        "window",
        "a",
        "b",
        "beta",
        "se",
        "n_obs",
        "r2",
    }
    missing = required.difference(win.columns)
    if missing:
        raise ValueError(f"WINDOWS_ALL_PATH missing columns: {missing}")

    # Add m∈[13,24] "post_yr2" window
    win = add_post_year2_windows(win)

    # t-stat and (approximate) normal p-value
    win["t_stat"] = win["beta"] / win["se"]
    win["p_norm"] = win["t_stat"].apply(normal_p_two_sided)
    win["stars_norm"] = win["p_norm"].apply(stars_from_p)

    # Simple 95% CI (normal approx)
    z_crit = 1.96
    win["ci_low_95"] = win["beta"] - z_crit * win["se"]
    win["ci_high_95"] = win["beta"] + z_crit * win["se"]

    out_path = TABLES_OUTPUT_DIR / "model1a_lp_windows_for_tables.tsv"
    win.to_csv(out_path, sep="\t", index=False)
    print(f"Saved window-level helper file to: {out_path}")

    return win


def build_dynamic_for_appendix() -> pd.DataFrame:
    """\
    Take model1a_lp_dynamic_betas_all.tsv and:

      * add stars based on the cluster-robust p-value column "pvalue",
      * carry through the n_event_obs counts as "N_event".

    Output:
      Design/Output (new)/Model_1A/Tables/model1a_lp_dynamic_for_appendix.tsv
    """
    dyn = pd.read_csv(DYNAMIC_ALL_PATH, sep="\t")

    # Harmonize "t" -> "tvalue" if needed
    if "t" in dyn.columns and "tvalue" not in dyn.columns:
        dyn = dyn.rename(columns={"t": "tvalue"})

    required = {
        "reform",
        "target",
        "spec_name",
        "event_time",
        "beta",
        "se",
        "tvalue",
        "pvalue",
        "n_event_obs",
        "n_obs",
        "r2",
    }
    missing = required.difference(dyn.columns)
    if missing:
        raise ValueError(f"DYNAMIC_ALL_PATH missing columns: {missing}")

    dyn["stars_cluster"] = dyn["pvalue"].apply(stars_from_p)
    dyn["N_event"] = dyn["n_event_obs"]

    out_path = TABLES_OUTPUT_DIR / "model1a_lp_dynamic_for_appendix.tsv"
    dyn.to_csv(out_path, sep="\t", index=False)
    print(f"Saved dynamic-event-time helper file to: {out_path}")

    return dyn


def build_pretrend_for_tables() -> pd.DataFrame:
    """\
    Take model1a_lp_pretrend_tests_all.tsv and just pass through the
    columns produced by Model_1A.py (with your actual naming).

    Output:
      Design/Output (new)/Model_1A/Tables/model1a_lp_pretrend_for_tables.tsv
    """
    pre = pd.read_csv(PRETREND_PATH, sep="\t")

    # Your actual columns (checked from the file):
    # ['reform', 'target', 'spec', 'pre_min', 'pre_max',
    #  'n_leads_total', 'n_bins_defined', 'n_bins_used',
    #  'f_stat', 'pvalue', 'df_num', 'df_denom', 'n_obs', 'r2']

    required = {
        "reform",
        "target",
        "spec",
        "pre_min",
        "pre_max",
        "f_stat",
        "pvalue",
        "df_num",
        "df_denom",
        "n_obs",
        "r2",
    }
    missing = required.difference(pre.columns)
    if missing:
        raise ValueError(f"PRETREND_PATH missing columns: {missing}")

    out_path = TABLES_OUTPUT_DIR / "model1a_lp_pretrend_for_tables.tsv"
    pre.to_csv(out_path, sep="\t", index=False)
    print(f"Saved pretrend helper file to: {out_path}")

    return pre


def main() -> None:
    print("=== Model_1A_to_tables: starting ===")
    print(f"THESIS_ROOT: {THESIS_ROOT}")
    print(f"Model 1A output dir: {MODEL1A_OUTPUT_DIR}")
    print(f"Tables output dir: {TABLES_OUTPUT_DIR}")

    win = build_windows_for_tables()
    dyn = build_dynamic_for_appendix()
    pre = build_pretrend_for_tables()

    print("\nSummary:")
    print(f"  Windows rows: {len(win)}")
    print(f"  Dynamic rows: {len(dyn)}")
    print(f"  Pretrend rows: {len(pre)}")
    print(f"Table helper files written to: {TABLES_OUTPUT_DIR}")
    print("=== Model_1A_to_tables: done ===")


if __name__ == "__main__":
    main()
