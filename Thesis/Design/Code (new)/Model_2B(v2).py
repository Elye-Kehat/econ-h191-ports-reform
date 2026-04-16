#!/usr/bin/env python3
"""
Model_2B(v2).py

Port/cluster-level regressions of ln(LP) on ln(K/L),
with separate specifications for each depreciation case.

v2 changes (important):
- Supports numeric depreciation column `delta` (e.g., 0.04/0.06/0.08).
  If 'dep_scenario' is missing but 'delta' exists, we derive dep_scenario
  labels from delta (low/central/high when it matches 0.04/0.06/0.08).
- If 'dep_scenario' exists but 'delta' is missing, we try to map
  dep_scenario -> delta (low/central/high) where possible.
- If neither exists, we assume central (dep_scenario='central', delta=0.06).
- Coerces numeric columns and drops inf/NaN rows in log_LP/log_KL/t_index
  before fitting.
- Carries `delta` into the tidy regression output.

Input (default):
    - Design/Output (new)/Model_2B/model2b_cluster_panel.tsv  (preferred)
      or, if missing:
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


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _dep_scenario_from_delta(delta_val: float) -> str:
    if not np.isfinite(delta_val):
        return "unknown"
    d = round(float(delta_val), 2)
    if abs(d - 0.04) < 1e-9:
        return "low"
    if abs(d - 0.06) < 1e-9:
        return "central"
    if abs(d - 0.08) < 1e-9:
        return "high"
    return f"delta_{d:.2f}"


def _delta_from_dep_scenario(s: str) -> float:
    if not isinstance(s, str):
        return np.nan
    k = s.strip().lower()
    if k in {"low", "lo"}:
        return 0.04
    if k in {"central", "center", "mid", "baseline"}:
        return 0.06
    if k in {"high", "hi"}:
        return 0.08
    return np.nan


def load_cluster_panel(panel_path: Path) -> pd.DataFrame:
    print(f"[Model 2B] Loading port/cluster panel from: {panel_path}")
    df = pd.read_csv(panel_path, sep="\t")

    # Coerce core numeric columns early
    df = _coerce_numeric(df, ["log_LP", "log_KL", "delta", "month_index", "t_index"])

    has_delta = "delta" in df.columns and df["delta"].notna().any()
    has_dep = "dep_scenario" in df.columns and df["dep_scenario"].notna().any()

    if not has_dep and not has_delta:
        print(
            "[Model 2B] WARNING: neither 'dep_scenario' nor 'delta' found. "
            "Assuming central depreciation (dep_scenario='central', delta=0.06)."
        )
        df["dep_scenario"] = "central"
        df["delta"] = 0.06

    elif not has_dep and has_delta:
        print("[Model 2B] 'dep_scenario' missing but 'delta' present; deriving dep_scenario from delta.")
        df["delta"] = pd.to_numeric(df["delta"], errors="coerce").round(2)
        df["dep_scenario"] = df["delta"].apply(_dep_scenario_from_delta)

    elif has_dep and not has_delta:
        print("[Model 2B] 'delta' missing but 'dep_scenario' present; attempting to map dep_scenario -> delta.")
        df["delta"] = df["dep_scenario"].apply(_delta_from_dep_scenario)

    else:
        df["delta"] = pd.to_numeric(df["delta"], errors="coerce").round(2)

    required_cols = ["log_LP", "log_KL", "dep_scenario"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[Model 2B] Expected column '{col}' in panel, but it is missing.")

    # Identify time index
    if "t_index" in df.columns and df["t_index"].notna().any():
        time_col = "t_index"
    elif "month_index" in df.columns and df["month_index"].notna().any():
        time_col = "month_index"
    else:
        raise ValueError(
            "[Model 2B] Could not find a time index column (looked for 't_index' or 'month_index')."
        )

    df["t_index"] = pd.to_numeric(df[time_col], errors="coerce")

    # Drop rows missing core variables
    before = len(df)
    df = df.dropna(subset=["log_LP", "log_KL", "t_index", "dep_scenario"])
    after = len(df)
    if after < before:
        print(f"[Model 2B] Dropped {before - after:,} rows with missing core regression variables.")

    scen_vals = sorted(df["dep_scenario"].unique())
    delta_vals = sorted([float(x) for x in df["delta"].dropna().unique()]) if "delta" in df.columns else []
    print(f"[Model 2B] Loaded {len(df):,} rows; time index column = '{time_col}'.")
    print(f"[Model 2B] dep_scenario values: {scen_vals}")
    if delta_vals:
        print(f"[Model 2B] delta values (rounded): {delta_vals}")

    return df


def build_spec_configs() -> list[dict]:
    """
    Define time-series specs for Model 2B.

    Baseline: ln(LP) on ln(K/L) + linear time trend over the full sample.
    """
    return [
        dict(
            spec_id="2B_ts_trend",
            description="Cluster ln(LP) on ln(K/L) with linear time trend (full sample).",
            sample_query=None,  # full sample
            formula="log_LP ~ log_KL + t_index",
        )
    ]


def fit_one_spec(df: pd.DataFrame, formula: str):
    """
    Run OLS with HC1 SEs (TS with one or few series: clustering usually not helpful).
    """
    res = smf.ols(formula, data=df).fit(cov_type="HC1")
    return res, len(df)


def run_model_2b(panel_path: Path, out_path: Path) -> None:
    df = load_cluster_panel(panel_path)
    specs = build_spec_configs()

    scenarios = sorted(df["dep_scenario"].unique())
    print(f"[Model 2B] Will run specs for dep_scenario values: {scenarios}")

    rows = []

    for scenario in scenarios:
        df_s = df[df["dep_scenario"] == scenario].copy()
        if df_s.empty:
            print(f"[Model 2B] WARNING: no rows for dep_scenario == '{scenario}'. Skipping.")
            continue

        # delta metadata (if present)
        delta_val = np.nan
        if "delta" in df_s.columns and df_s["delta"].notna().any():
            uniq = sorted(df_s["delta"].dropna().unique())
            delta_val = float(uniq[0]) if len(uniq) == 1 else float(np.nan)

        print(
            f"[Model 2B] Running specs for dep_scenario = '{scenario}' "
            f"({df_s.shape[0]:,} obs)"
            + (f", delta={delta_val:.2f}" if np.isfinite(delta_val) else "")
            + "."
        )

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
                        delta=delta_val,  # may be NaN if heterogeneous/missing
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
        help=("Path to cluster/port panel TSV. If not provided, defaults to "
              "ThesisRoot/Design/Output (new)/Model_2B/model2b_cluster_panel.tsv "
              "(or model2b_port_panel.tsv if cluster panel is missing)."),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=("Path for output TSV of regression results. If not provided, defaults to "
              "ThesisRoot/Design/Output (new)/Model_2B/model2b_reg_results.tsv"),
    )
    args = parser.parse_args()

    thesis_root = find_thesis_root()
    print(f"[Model 2B] Thesis root: {thesis_root}")

    if args.panel is None:
        candidate_cluster = thesis_root / "Design" / "Output (new)" / "Model_2B" / "model2b_cluster_panel.tsv"
        candidate_port = thesis_root / "Design" / "Output (new)" / "Model_2B" / "model2b_port_panel.tsv"
        panel_path = candidate_cluster if candidate_cluster.exists() else candidate_port
    else:
        panel_path = args.panel

    if args.out is None:
        out_path = thesis_root / "Design" / "Output (new)" / "Model_2B" / "model2b_reg_results.tsv"
    else:
        out_path = args.out

    run_model_2b(panel_path, out_path)


if __name__ == "__main__":
    main()



# ----------------------------------------------------------------------
# DIAGNOSTIC NOTE (Model_2B(v2) output + interpretation)
#
# v2 behavior (intended):
# - If dep_scenario is missing but numeric delta exists (0.04/0.06/0.08),
#   we derive dep_scenario ∈ {low, central, high} from delta and carry delta
#   through to the tidy regression output.
# - In the current cluster panel, this yields 105 rows total:
#     3 scenarios × 35 monthly observations each.
#
# Regression estimated here (per scenario):
#   log_LP ~ log_KL + t_index
# with HC1 robust SEs (time-series setting; clustering is not used here).
#
# Expected tidy output structure:
# - model2b_reg_results.tsv has 9 rows:
#     3 scenarios × 3 parameters (Intercept, log_KL, t_index).
# - Each scenario row block carries n_obs and r2 for that fit.
#
# Statistical limitation (not a code bug):
# - HC1 addresses heteroskedasticity but not serial correlation.
# - If residuals are autocorrelated (likely in monthly time series), consider
#   HAC/Newey–West SEs later when inference matters and once true monthly
#   labor-hours (LP) data is available.
#
# Substantive note:
# - Coefficients may look very similar across deltas; this can happen because
#   delta changes constructed K/L only modestly relative to the variation in
#   the short cluster time series.
# ----------------------------------------------------------------------