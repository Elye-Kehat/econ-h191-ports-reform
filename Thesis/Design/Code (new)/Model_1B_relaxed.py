"""
Model_1B_relaxed.py

"Relaxed" K/L event-study for Haifa:

  * Reuses Model_1B's K/L ES sample construction (build_base_specs_kl,
    build_es_sample_kl), so event-time definitions and windows match
    Model_1B exactly.

  * Relaxes the FE structure to avoid saturation and make inference usable
    with the limited K/L data:

      log_KL_it = α_i + δ * t_index_t
                  + Σ_m β_m * 1{unit i is treated & event_time_t = m} + ε_it,

    where:
      - i indexes series (Haifa-Legacy terminal vs Haifa port control),
      - t_index is a linear time index (common trend),
      - β_m are treated-only event-time coefficients, with m = -1 omitted
        as the reference period.

  * Uses HC1 heteroskedastic-robust standard errors (no clustering).
  * Aggregates β_m into window-average effects using the delta method for
    95% SE (normal approximation):

        pre_all:  m ∈ [-12, -2]
        post_y1:  m ∈ [  1, 12]
        post_y2:  m ∈ [ 13, 24]
        post_all: m ∈ [  1, 24]

Output:
  Design/Output (new)/Model_1B_relaxed/model1b_relaxed_window_betas.tsv
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import Model_1B as m1b  # reuse ES sample construction for K/L


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
THESIS_ROOT = THIS_FILE.parents[2]

KL_PANEL_PATH = THESIS_ROOT / "Data" / "KL" / "KL_Panel_monthly.tsv"

OUTPUT_DIR = THESIS_ROOT / "Design" / "Output (new)" / "Model_1B_relaxed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_BETAS_PATH = OUTPUT_DIR / "model1b_relaxed_window_betas.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normal_p_two_sided(t: float) -> float:
    """Approximate two-sided p-value under N(0,1) for a t-statistic."""
    if not np.isfinite(t):
        return np.nan
    z = abs(float(t))
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return 2.0 * (1.0 - cdf)


def make_month_index(df_kl: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], int]]:
    """
    Build a global month_index exactly as in Model_1B, so that
    build_es_sample_kl() behaves consistently.
    """
    df = df_kl.copy()
    df["ym_tuple"] = list(zip(df["year"], df["month"]))
    ym_sorted = sorted(df["ym_tuple"].unique())
    ym_to_idx = {ym: i + 1 for i, ym in enumerate(ym_sorted)}
    df["month_index"] = df["ym_tuple"].map(ym_to_idx)
    return df, ym_to_idx


def make_et_col_name(m: int) -> str:
    """
    Construct a valid Python/column name for event time m.
    Example:
      m = -12 -> "et_mneg12"
      m = 1   -> "et_mpos1"
    """
    if m < 0:
        return f"et_mneg{abs(m)}"
    else:
        return f"et_mpos{m}"


# Windows: match Model_1B's K/L window definitions
WINDOW_DEFS: List[Tuple[str, int, int]] = [
    ("post_all", 1, 24),
    ("post_y1", 1, 12),
    ("post_y2", 13, 24),
    ("pre_all", -12, -2),
]


# ---------------------------------------------------------------------------
# Core relaxed ES regression
# ---------------------------------------------------------------------------


def run_es_relaxed_for_spec(
    df_es: pd.DataFrame,
    reform: str,
    target: str,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run the relaxed ES regression for one (reform, target) spec.

    df_es is produced by Model_1B.build_es_sample_kl and contains:
      - log_KL, series_id, month_index, treat, event_time, ...
    """

    if df_es is None or df_es.empty:
        return [], []

    df = df_es.copy()

    # Linear time index (common trend)
    t0 = df["month_index"].min()
    df["t_index"] = df["month_index"] - t0

    # Treated-only event-time dummies (omit m = -1 as reference)
    treated_mask = df["treat"] == 1
    event_grid = sorted(df.loc[treated_mask, "event_time"].unique())

    et_cols: List[str] = []
    et_map: Dict[str, int] = {}

    for m in event_grid:
        if m == -1:
            continue  # reference period
        col = make_et_col_name(m)
        df[col] = ((df["treat"] == 1) & (df["event_time"] == m)).astype(int)
        et_cols.append(col)
        et_map[col] = int(m)

    if not et_cols:
        print(f"[WARN] Relaxed ES: no event-time support for {reform} – {target}; skipping.")
        return [], []

    # Build formula: log_KL ~ series FE + common linear trend + treated-only ET dummies
    rhs_terms: List[str] = ["C(series_id)", "t_index"]
    rhs_terms.extend(et_cols)
    formula = "log_KL ~ " + " + ".join(rhs_terms)

    print(f"  [relaxed] OLS formula for {reform} – {target}: {formula}")

    model = smf.ols(formula, data=df)
    # Use HC1 heteroskedastic-robust covariance (no clustering)
    result = model.fit(cov_type="HC1")

    n_obs = int(result.nobs)
    treated_n = int(df.loc[df["treat"] == 1, "log_KL"].notna().sum())
    control_n = int(df.loc[df["treat"] == 0, "log_KL"].notna().sum())

    print(
        f"    n_obs={n_obs}, treated_n={treated_n}, control_n={control_n}, "
        f"R^2={result.rsquared:.3f}, df_resid={result.df_resid:.1f}"
    )

    # ----------------------------------------------------------------------
    # Dynamic rows (one per m), in case we need them later
    # ----------------------------------------------------------------------
    dynamic_rows: List[Dict] = []
    params = result.params
    bse = result.bse
    tvals = result.tvalues
    pvals = result.pvalues
    cov_type = result.cov_type
    r2 = float(result.rsquared)

    for col in et_cols:
        m = et_map[col]
        beta = float(params.get(col, np.nan))
        se = float(bse.get(col, np.nan))
        tval = float(tvals.get(col, np.nan))
        pval = float(pvals.get(col, np.nan))

        # N(m): number of treated observations at this event time
        N_m = int(df.loc[treated_mask & (df["event_time"] == m), "log_KL"].notna().sum())

        dynamic_rows.append(
            {
                "model": "1B_relaxed",
                "reform": reform,
                "target": target,
                "fe_type": "relaxed",
                "event_time": m,
                "beta_hat": beta,
                "se": se,
                "tvalue": tval,
                "pvalue": pval,
                "N_m": N_m,
                "n_obs": n_obs,
                "treated_n": treated_n,
                "control_n": control_n,
                "cov_type": cov_type,
                "cluster_by": "series_id",  # conceptually, but we used HC1
                "r2": r2,
            }
        )

    # ----------------------------------------------------------------------
    # Window-average rows (delta method)
    # ----------------------------------------------------------------------
    window_rows: List[Dict] = []
    cov = result.cov_params()

    # map event_time -> column for quick lookup
    et_by_m: Dict[int, str] = {m: col for col, m in et_map.items()}

    for window_name, m_start, m_end in WINDOW_DEFS:
        # Which event-time dummies fall inside this window?
        ms_in_window = [
            m for m in et_by_m.keys() if (m_start <= m <= m_end and m != -1)
        ]
        if not ms_in_window:
            # No support in this window; skip
            print(
                f"    [relaxed] Window '{window_name}' has no event-times "
                f"for {reform} – {target}; skipping."
            )
            continue

        cols = [et_by_m[m] for m in sorted(ms_in_window)]

        # Coefficient vector and covariance submatrix
        b_vec = params[cols].values.astype(float)
        cov_sub = cov.loc[cols, cols].values.astype(float)

        k = len(cols)
        weights = np.full(k, 1.0 / k, dtype=float)

        beta_bar = float(weights @ b_vec)
        var_bar = float(weights @ cov_sub @ weights)

        if var_bar < 0:
            # Numerical guard; shouldn't happen often
            var_bar = float("nan")

        se_bar = math.sqrt(var_bar) if np.isfinite(var_bar) else float("nan")
        if np.isfinite(se_bar) and se_bar > 0:
            t_bar = beta_bar / se_bar
            p_bar = normal_p_two_sided(t_bar)
        else:
            t_bar = float("nan")
            p_bar = float("nan")


        m_start_eff = int(min(ms_in_window)) if ms_in_window else m_start
        m_end_eff = int(max(ms_in_window)) if ms_in_window else m_end

        window_rows.append(
            {
                "model": "1B_relaxed",
                "reform": reform,
                "target": target,
                "fe_type": "relaxed",
                "window_name": window_name,
                "m_start": m_start_eff,
                "m_end": m_end_eff,
                "beta_hat": beta_bar,
                "se": se_bar,
                "tvalue": t_bar,
                "pvalue": p_bar,
                "n_obs": n_obs,
                "treated_n": treated_n,
                "control_n": control_n,
                "cov_type": cov_type,
                "cluster_by": "series_id",
                "r2": r2,
            }
        )

    return dynamic_rows, window_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== Model_1B_relaxed (K/L) event-study: starting ===")
    print(f"Thesis root: {THESIS_ROOT}")
    print(f"K/L panel path: {KL_PANEL_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    df_kl = pd.read_csv(KL_PANEL_PATH, sep="\t")
    print(f"Loaded {len(df_kl)} rows from KL_Panel_monthly.tsv.")

    df_kl, ym_to_idx = make_month_index(df_kl)

    # Use the same base specs as Model_1B (haifa_comp and haifa_priv on K/L)
    base_specs = m1b.build_base_specs_kl()
    print(f"Number of base K/L specs: {len(base_specs)}")

    all_dyn_rows: List[Dict] = []
    all_win_rows: List[Dict] = []

    for spec in base_specs:
        print(f"--- Relaxed ES for {spec.reform} – {spec.target} ---")
        df_es = m1b.build_es_sample_kl(df_kl.copy(), spec, ym_to_idx)
        if df_es is None or df_es.empty:
            print(
                f"  [WARN] No ES sample returned for {spec.reform} – {spec.target}; skipping."
            )
            continue

        dyn_rows, win_rows = run_es_relaxed_for_spec(
            df_es=df_es,
            reform=spec.reform,
            target=spec.target,
        )
        all_dyn_rows.extend(dyn_rows)
        all_win_rows.extend(win_rows)

    # Assemble DataFrames
    if all_win_rows:
        win_df = pd.DataFrame(all_win_rows)
    else:
        win_df = pd.DataFrame(
            columns=[
                "model",
                "reform",
                "target",
                "fe_type",
                "window_name",
                "m_start",
                "m_end",
                "beta_hat",
                "se",
                "tvalue",
                "pvalue",
                "n_obs",
                "treated_n",
                "control_n",
                "cov_type",
                "cluster_by",
                "r2",
            ]
        )

    # We only *need* window betas for now
    win_df.to_csv(WINDOW_BETAS_PATH, sep="\t", index=False)
    print(f"Wrote relaxed K/L window betas to: {WINDOW_BETAS_PATH}")
    print(f"Total relaxed window rows: {len(win_df)}")

    print("=== Model_1B_relaxed: done ===")


if __name__ == "__main__":
    main()
