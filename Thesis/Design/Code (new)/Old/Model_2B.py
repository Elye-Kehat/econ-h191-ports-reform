#!/usr/bin/env python3
"""
Model_2B.py

Port/cluster-level regressions of ln(LP) on ln(K/L),
with separate specifications for each depreciation scenario (low/central/high).

If the input panel does NOT contain a 'dep_scenario' column, we assume
everything is the central depreciation case and set dep_scenario = 'central'.

Input:
    - Design/Output (new)/Model_2B/model2b_port_panel.tsv

Output:
    - Design/Output (new)/Model_2B/model2b_reg_results.tsv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def find_thesis_root(start: Path | None = None) -> Path:
    here = start or Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "Data").exists() and (p / "Design").exists():
            return p
    raise RuntimeError("Could not find thesis root (no Data/ and Design/ siblings found).")


def load_port_panel(panel_path: Path) -> pd.DataFrame:
    print(f"[Model 2B] Loading port/cluster panel from: {panel_path}")
    df = pd.read_csv(panel_path, sep="\t")

    # If dep_scenario is missing, assume everything is central for now.
    if "dep_scenario" not in df.columns:
        print("[Model 2B] WARNING: 'dep_scenario' column not found. "
              "Assuming all observations are central depreciation (dep_scenario = 'central').")
        df["dep_scenario"] = "central"

    required_cols = ["log_LP", "log_KL", "dep_scenario"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[Model 2B] Expected column '{col}' in panel, but it is missing.")

    # Identify time index
    if "t_index" in df.columns:
        time_col = "t_index"
    elif "month_index" in df.columns:
        time_col = "month_index"
    else:
        raise ValueError(
            "[Model 2B] Could not find a time index column "
            "(looked for 't_index' or 'month_index')."
        )

    df["t_index"] = df[time_col]

    print(f"[Model 2B] Loaded {len(df):,} rows; time index column = '{time_col}'.")
    print(f"[Model 2B] dep_scenario values: {sorted(df['dep_scenario'].unique())}")
    return df


def build_spec_configs() -> list[dict]:
    """
    Define time-series specs for Model 2B.

    Baseline here: ln(LP) on ln(K/L) + linear time trend over the full sample.
    """
    specs = []

    specs.append(
        dict(
            spec_id="2B_ts_trend",
            description="Cluster ln(LP) on ln(K/L) with linear time trend (full sample).",
            sample_query=None,  # full sample
            formula="log_LP ~ log_KL + t_index",
        )
    )

    return specs


def fit_one_spec(df: pd.DataFrame, formula: str) -> tuple:
    """
    Run OLS with HC1 SEs (TS with one or few series: clustering usually not helpful).
    """
    res = smf.ols(formula, data=df).fit(cov_type="HC1")
    return res, len(df)


def run_model_2b(panel_path: Path, out_path: Path) -> None:
    df = load_port_panel(panel_path)
    specs = build_spec_configs()

    # Auto-detect which depreciation scenarios exist in the data
    scenarios = sorted(df["dep_scenario"].unique())
    print(f"[Model 2B] Will run specs for dep_scenario values: {scenarios}")

    rows = []

    for scenario in scenarios:
        df_s = df[df["dep_scenario"] == scenario].copy()
        if df_s.empty:
            print(f"[Model 2B] WARNING: no rows for dep_scenario == '{scenario}'. Skipping.")
            continue

        print(f"[Model 2B] Running specs for dep_scenario = '{scenario}' "
              f"({df_s.shape[0]:,} obs).")

        for spec in specs:
            spec_id = spec["spec_id"]
            formula = spec["formula"]
            sample_query = spec.get("sample_query")

            if sample_query:
                df_spec = df_s.query(sample_query).copy()
            else:
                df_spec = df_s.copy()

            if df_spec.empty:
                print(f"[Model 2B]   Spec {spec_id}: empty sample after filtering. Skipping.")
                continue

            print(f"[Model 2B]   Spec {spec_id}: running OLS on {len(df_spec):,} rows.")
            res, n_obs = fit_one_spec(df_spec, formula)

            for param, beta in res.params.items():
                se = res.bse.get(param, np.nan)
                tval = res.tvalues.get(param, np.nan)
                pval = res.pvalues.get(param, np.nan)

                rows.append(
                    dict(
                        model="2B",
                        spec_id=spec_id,
                        dep_scenario=scenario,
                        param=param,
                        beta=beta,
                        se=se,
                        tvalue=tval,
                        pvalue=pval,
                        n_obs=n_obs,
                        r2=res.rsquared,
                        dep_var="log_LP",
                        formula=formula,
                        description=spec.get("description", ""),
                    )
                )

    if not rows:
        raise RuntimeError("[Model 2B] No regression results produced; check dep_scenario labels and spec filters.")

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"[Model 2B] Wrote tidy regression results to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Model 2B: port/cluster ln(LP) ~ ln(K/L) by depreciation scenario.")
    parser.add_argument(
        "--panel",
        type=Path,
        default=None,
        help=("Path to model2b_port_panel.tsv (TSV, tab-separated). "
              "If not provided, defaults to "
              "ThesisRoot/Design/Output (new)/Model_2B/model2b_port_panel.tsv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=("Path for output TSV of regression results. "
              "If not provided, defaults to "
              "ThesisRoot/Design/Output (new)/Model_2B/model2b_reg_results.tsv"),
    )
    args = parser.parse_args()

    thesis_root = find_thesis_root()
    print(f"[Model 2B] Thesis root: {thesis_root}")

    if args.panel is None:
        panel_path = thesis_root / "Design" / "Output (new)" / "Model_2B" / "model2b_port_panel.tsv"
    else:
        panel_path = args.panel

    if args.out is None:
        out_path = thesis_root / "Design" / "Output (new)" / "Model_2B" / "model2b_reg_results.tsv"
    else:
        out_path = args.out

    run_model_2b(panel_path, out_path)


if __name__ == "__main__":
    main()
