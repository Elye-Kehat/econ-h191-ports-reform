#!/usr/bin/env python3
"""
Model_2A(v2).py

Terminal-level regressions of ln(LP) on ln(K/L),
with separate specifications for each depreciation case.

v2 change (important):
- Supports numeric depreciation column `delta` (e.g., 0.04/0.06/0.08).
  If 'dep_scenario' is missing but 'delta' exists, we derive dep_scenario
  labels from delta (low/central/high when it matches 0.04/0.06/0.08).
- If 'dep_scenario' exists but 'delta' is missing, we try to map
  dep_scenario -> delta (low/central/high) where possible.
- Always carries `delta` into the tidy regression output (when available).
- Cleans numeric columns and drops inf/NaN rows in log_LP/log_KL before fitting.

Input:
    - Design/Output (new)/Model_2A/model2a_terminal_panel.tsv

Output:
    - Design/Output (new)/Model_2A/model2a_reg_results.tsv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def find_thesis_root(start: Path | None = None) -> Path:
    """
    Walk up from the script location (or a provided path) until we find a directory
    that looks like the thesis root (contains Data/ and Design/).
    """
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
    """
    Map numeric delta to canonical scenario labels when it matches common values.
    Otherwise return a stable label.
    """
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
    """
    Map dep_scenario strings to numeric deltas when possible.
    """
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


def load_terminal_panel(panel_path: Path) -> pd.DataFrame:
    print(f"[Model 2A] Loading terminal panel from: {panel_path}")
    df = pd.read_csv(panel_path, sep="\t")

    # Coerce core numeric columns early
    df = _coerce_numeric(df, ["log_LP", "log_KL", "delta", "month_index", "t_index"])

    # Handle depreciation dimension:
    # Priority: if delta exists, derive dep_scenario. Otherwise use dep_scenario.
    has_delta = "delta" in df.columns and df["delta"].notna().any()
    has_dep = "dep_scenario" in df.columns and df["dep_scenario"].notna().any()

    if not has_dep and not has_delta:
        print(
            "[Model 2A] WARNING: neither 'dep_scenario' nor 'delta' found. "
            "Assuming central depreciation (dep_scenario='central', delta=0.06)."
        )
        df["dep_scenario"] = "central"
        df["delta"] = 0.06

    elif not has_dep and has_delta:
        print("[Model 2A] 'dep_scenario' missing but 'delta' present; deriving dep_scenario from delta.")
        df["delta"] = pd.to_numeric(df["delta"], errors="coerce").round(2)
        df["dep_scenario"] = df["delta"].apply(_dep_scenario_from_delta)

    elif has_dep and not has_delta:
        print("[Model 2A] 'delta' missing but 'dep_scenario' present; attempting to map dep_scenario -> delta.")
        df["delta"] = df["dep_scenario"].apply(_delta_from_dep_scenario)

    else:
        # both exist: keep, but normalize delta
        df["delta"] = pd.to_numeric(df["delta"], errors="coerce").round(2)

    required_cols = ["log_LP", "log_KL", "dep_scenario"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[Model 2A] Expected column '{col}' in panel, but it is missing.")

    # Identify a terminal / unit id column for FE
    if "terminal_id" in df.columns:
        unit_col = "terminal_id"
    elif "series_id" in df.columns:
        unit_col = "series_id"
    elif "terminal" in df.columns:
        unit_col = "terminal"
    else:
        raise ValueError(
            "[Model 2A] Could not find a terminal identifier column "
            "(looked for 'terminal_id', 'series_id', 'terminal')."
        )

    # Identify a time index column for time FE
    if "month_index" in df.columns and df["month_index"].notna().any():
        time_col = "month_index"
    elif "t_index" in df.columns and df["t_index"].notna().any():
        time_col = "t_index"
    else:
        raise ValueError(
            "[Model 2A] Could not find a time index column "
            "(looked for 'month_index' or 't_index')."
        )

    df["unit_id"] = df[unit_col]
    df["time_fe"] = df[time_col]

    # Drop rows missing core regression variables
    before = len(df)
    df = df.dropna(subset=["log_LP", "log_KL", "unit_id", "time_fe", "dep_scenario"])
    after = len(df)
    if after < before:
        print(f"[Model 2A] Dropped {before - after:,} rows with missing core regression variables.")

    scen_vals = sorted(df["dep_scenario"].unique())
    delta_vals = sorted([d for d in df["delta"].dropna().unique()]) if "delta" in df.columns else []
    print(f"[Model 2A] Loaded {len(df):,} rows; unit FE column = '{unit_col}', time FE column = '{time_col}'.")
    print(f"[Model 2A] dep_scenario values: {scen_vals}")
    if delta_vals:
        print(f"[Model 2A] delta values (rounded): {delta_vals}")

    return df


def build_spec_configs() -> list[dict]:
    """
    Define which samples / formulas we run for Model 2A.
    Currently: one baseline spec over the full sample.

    Add more entries here with different 'sample_query' strings if needed.
    """
    specs = []

    # Baseline: full sample, ln(LP) on ln(K/L) + unit FE + time FE
    specs.append(
        dict(
            spec_id="2A_baseline",
            description="Terminal ln(LP) on ln(K/L) with unit and time FE (full sample)",
            sample_query=None,  # None → use all rows for that dep_scenario
            formula="log_LP ~ log_KL + C(unit_id) + C(time_fe)",
        )
    )

    return specs


def fit_one_spec(df: pd.DataFrame, formula: str) -> tuple:
    """
    Run OLS with heteroskedasticity-robust SEs.
    Clustered SEs by unit_id if there is more than one unit.
    """
    if "unit_id" in df.columns and df["unit_id"].nunique() > 1:
        res = smf.ols(formula, data=df).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["unit_id"]},
        )
    else:
        res = smf.ols(formula, data=df).fit(cov_type="HC1")

    return res, len(df)


def run_model_2a(panel_path: Path, out_path: Path) -> None:
    df = load_terminal_panel(panel_path)
    specs = build_spec_configs()

    # Auto-detect which depreciation scenarios exist in the data
    scenarios = sorted(df["dep_scenario"].unique())
    print(f"[Model 2A] Will run specs for dep_scenario values: {scenarios}")

    rows = []

    for scenario in scenarios:
        df_s = df[df["dep_scenario"] == scenario].copy()
        if df_s.empty:
            print(f"[Model 2A] WARNING: no rows for dep_scenario == '{scenario}'. Skipping.")
            continue

        # delta metadata (if present)
        delta_val = np.nan
        if "delta" in df_s.columns and df_s["delta"].notna().any():
            uniq = sorted(df_s["delta"].dropna().unique())
            delta_val = float(uniq[0]) if len(uniq) == 1 else float(np.nan)

        print(
            f"[Model 2A] Running specs for dep_scenario = '{scenario}' "
            f"({df_s['unit_id'].nunique()} terminals, {len(df_s):,} obs)"
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
                print(f"[Model 2A]   Spec {spec_id}: empty sample after filtering. Skipping.")
                continue

            print(f"[Model 2A]   Spec {spec_id}: running OLS on {len(df_spec):,} rows.")
            res, n_obs = fit_one_spec(df_spec, formula)

            # Store all coefficients in tidy form
            for param, beta in res.params.items():
                se = res.bse.get(param, np.nan)
                tval = res.tvalues.get(param, np.nan)
                pval = res.pvalues.get(param, np.nan)

                rows.append(
                    dict(
                        model="2A",
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
        raise RuntimeError("[Model 2A] No regression results produced; check scenario labels and spec filters.")

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"[Model 2A] Wrote tidy regression results to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Model 2A: terminal ln(LP) ~ ln(K/L) by depreciation scenario.")
    parser.add_argument(
        "--panel",
        type=Path,
        default=None,
        help=("Path to model2a_terminal_panel.tsv (TSV, tab-separated). "
              "If not provided, defaults to "
              "ThesisRoot/Design/Output (new)/Model_2A/model2a_terminal_panel.tsv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=("Path for output TSV of regression results. "
              "If not provided, defaults to "
              "ThesisRoot/Design/Output (new)/Model_2A/model2a_reg_results.tsv"),
    )
    args = parser.parse_args()

    thesis_root = find_thesis_root()
    print(f"[Model 2A] Thesis root: {thesis_root}")

    if args.panel is None:
        panel_path = thesis_root / "Design" / "Output (new)" / "Model_2A" / "model2a_terminal_panel.tsv"
    else:
        panel_path = args.panel

    if args.out is None:
        out_path = thesis_root / "Design" / "Output (new)" / "Model_2A" / "model2a_reg_results.tsv"
    else:
        out_path = args.out

    run_model_2a(panel_path, out_path)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# DIAGNOSTIC NOTE (Model_2A(v2) output + known limitations)
#
# v2 behavior (intended):
# - If dep_scenario is missing but numeric delta exists (0.04/0.06/0.08),
#   we derive dep_scenario ∈ {low, central, high} from delta and carry delta
#   through to the tidy regression output.
# - In the current panel, this produces 3 scenarios with 84 obs each
#   (2 terminals × 42 time periods).
#
# Regression run here:
#   log_LP ~ log_KL + C(unit_id) + C(time_fe)
# with clustered SEs by unit_id when >1 unit exists.
#
# IMPORTANT: small-cluster covariance pathology
# - Within each depreciation scenario there are only 2 clusters (two terminals).
# - Cluster-robust covariance with 2 clusters is numerically unstable in statsmodels
#   and can yield a non-PSD covariance matrix, triggering:
#     RuntimeWarning: invalid value encountered in sqrt
# - Practical consequences in the output TSV:
#     * some parameters' standard errors (including log_KL) may be NaN
#     * or SEs may be implausibly tiny (near-perfect-fit behavior)
# - Point estimates are still computed, but inference from these clustered SEs
#   should not be trusted in this tiny 2×T setting.
#
# Interpretation / usage:
# - Treat Model_2A(v2) as a "tidy FE regression output" utility, not the main
#   elasticity table used in the paper. The preferred η estimates are produced by
#   Model_2_to_tables(v2).py (HC1 + TS+trend and pooled FE specifications).
# - Once true monthly labor-hours data is incorporated and/or sample size expands,
#   revisit inference choices here (e.g., HC1, time clustering, CR2, or a different
#   specification) if we want Model_2A results to be used in tables.
# ----------------------------------------------------------------------