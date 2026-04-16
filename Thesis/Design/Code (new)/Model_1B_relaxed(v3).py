"""
Model_1B_relaxed(v3).py

Relaxed K/L event-study for Haifa.

Purpose:
  - Provide a non-saturated alternative for K/L event-time dynamics when the
    full TWFE + month FE design is numerically fragile in very small NYT samples.

Estimator (HC1):
  log_KL_it = α_i + δ*t_index_t + Σ_m β_m * 1{treated & event_time=m} + ε_it,
  with m=-1 omitted.

v3 updates:
  - Runs BOTH designs:
      * NYT  (design="nyt")  -> outputs model1b_relaxed_window_betas.tsv  (unchanged name)
      * TWFE (design="twfe") -> outputs model1b_relaxed_window_betas_twfe.tsv
  - Uses Model_1B(v3) sample construction (same event-time definition + NYT guard).
  - Minimal clutter: exports window betas only.

Output directory:
  Design/Output (new)/Model_1B_relaxed/
"""

from __future__ import annotations

import math
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------------
# Load Model_1B(v3) safely
# ---------------------------------------------------------------------------

MODEL_1B_FILENAME = "Model_1B(v3).py"
MODEL_1B_PATH = Path(__file__).with_name(MODEL_1B_FILENAME)
spec = importlib.util.spec_from_file_location("Model_1B_v3_relaxed_import", MODEL_1B_PATH)
m1b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m1b
assert spec.loader is not None
spec.loader.exec_module(m1b)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
THESIS_ROOT = THIS_FILE.parents[2]

KL_PANEL_PATH = THESIS_ROOT / "Data" / "KL" / "KL_Panel_monthly.tsv"

OUTPUT_DIR = THESIS_ROOT / "Design" / "Output (new)" / "Model_1B_relaxed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NYT_OUT = OUTPUT_DIR / "model1b_relaxed_window_betas.tsv"
TWFE_OUT = OUTPUT_DIR / "model1b_relaxed_window_betas_twfe.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normal_p_two_sided(t: float) -> float:
    if not np.isfinite(t):
        return np.nan
    z = abs(float(t))
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return 2.0 * (1.0 - cdf)


def make_month_index(df_kl: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], int]]:
    df = df_kl.copy()
    df["ym_tuple"] = list(zip(df["year"], df["month"]))
    ym_sorted = sorted(df["ym_tuple"].unique())
    ym_to_idx = {ym: i + 1 for i, ym in enumerate(ym_sorted)}
    df["month_index"] = df["ym_tuple"].map(ym_to_idx)
    return df, ym_to_idx


def make_et_col_name(m: int) -> str:
    return f"et_mneg{abs(m)}" if m < 0 else f"et_mpos{m}"


WINDOW_DEFS: List[Tuple[str, int, int]] = [
    ("post_all", 1, 24),
    ("post_y1", 1, 12),
    ("post_y2", 13, 24),
    ("pre_all", -12, -2),
]


def run_es_relaxed_for_spec(df_es: pd.DataFrame, reform: str, target: str, design: str) -> List[Dict]:
    if df_es is None or df_es.empty:
        return []

    df = df_es.copy()

    # Common linear time index
    t0 = df["month_index"].min()
    df["t_index"] = df["month_index"] - t0

    treated_mask = df["treat"] == 1
    event_grid = sorted(int(x) for x in df.loc[treated_mask, "event_time"].unique())

    et_cols: List[str] = []
    et_map: Dict[str, int] = {}

    for m in event_grid:
        if m == -1:
            continue
        col = make_et_col_name(m)
        df[col] = ((df["treat"] == 1) & (df["event_time"] == m)).astype(int)
        et_cols.append(col)
        et_map[col] = int(m)

    if not et_cols:
        print(f"[WARN] Relaxed ES: no event-time support for {reform} – {target} ({design}); skipping.")
        return []

    # Regression: series FE + linear trend + treated-only event-time dummies
    rhs = ["C(series_id)", "t_index"] + et_cols
    formula = "log_KL ~ " + " + ".join(rhs)

    print(f"  [relaxed:{design}] OLS formula for {reform} – {target}: {formula}")

    res = smf.ols(formula, data=df).fit(cov_type="HC1")

    n_obs = int(res.nobs)
    treated_n = int(df.loc[df["treat"] == 1, "log_KL"].notna().sum())
    control_n = int(df.loc[df["treat"] == 0, "log_KL"].notna().sum())
    r2 = float(res.rsquared)

    params = res.params
    cov = res.cov_params()

    # Map event_time -> dummy column
    et_by_m: Dict[int, str] = {m: col for col, m in et_map.items()}

    rows: List[Dict] = []
    for window_name, m_start, m_end in WINDOW_DEFS:
        ms_in_window = [m for m in et_by_m.keys() if (m_start <= m <= m_end and m != -1)]
        if not ms_in_window:
            continue

        cols = [et_by_m[m] for m in sorted(ms_in_window)]
        b_vec = params[cols].values.astype(float)
        cov_sub = cov.loc[cols, cols].values.astype(float)

        k = len(cols)
        w = np.full(k, 1.0 / k, dtype=float)

        beta_bar = float(w @ b_vec)
        var_bar = float(w @ cov_sub @ w)
        se_bar = math.sqrt(var_bar) if np.isfinite(var_bar) and var_bar >= 0 else float("nan")

        if np.isfinite(se_bar) and se_bar > 0:
            t_bar = beta_bar / se_bar
            p_bar = normal_p_two_sided(t_bar)
        else:
            t_bar = float("nan")
            p_bar = float("nan")

        rows.append(
            dict(
                model="1B_relaxed",
                design=design,
                reform=reform,
                target=target,
                fe_type="relaxed",
                window_name=window_name,
                m_start=int(min(ms_in_window)),
                m_end=int(max(ms_in_window)),
                beta_hat=beta_bar,
                se=se_bar,
                tvalue=t_bar,
                pvalue=p_bar,
                n_obs=n_obs,
                treated_n=treated_n,
                control_n=control_n,
                cov_type=res.cov_type,
                cluster_by="",
                r2=r2,
            )
        )

    return rows


def main() -> None:
    print("=== Model_1B_relaxed(v3): starting ===")
    print(f"Thesis root: {THESIS_ROOT}")
    print(f"KL panel path: {KL_PANEL_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    df_kl = pd.read_csv(KL_PANEL_PATH, sep="\t")
    print(f"Loaded {len(df_kl)} rows from KL_Panel_monthly.tsv.")

    # Ensure log_KL exists
    if "log_KL" not in df_kl.columns:
        if "KL" not in df_kl.columns:
            raise ValueError("KL_Panel_monthly.tsv must contain 'KL' or 'log_KL'.")
        df_kl["log_KL"] = np.log(df_kl["KL"])

    df_kl, ym_to_idx = make_month_index(df_kl)

    base_specs = m1b.build_base_specs_kl()
    print(f"Number of base K/L specs: {len(base_specs)}")

    # Run both designs with minimal outputs
    for design, out_path in [("nyt", NYT_OUT), ("twfe", TWFE_OUT)]:
        all_rows: List[Dict] = []
        print(f"\n==================== relaxed {design.upper()} ====================")

        for spec in base_specs:
            print(f"--- Relaxed ES ({design}) for {spec.reform} – {spec.target} ---")
            df_es = m1b.build_es_sample_kl(df_kl.copy(), spec, ym_to_idx, design=design)
            if df_es is None or df_es.empty:
                print(f"  [WARN] No ES sample for {spec.reform} – {spec.target} ({design}); skipping.")
                continue

            rows = run_es_relaxed_for_spec(df_es, spec.reform, spec.target, design)
            all_rows.extend(rows)

        out_df = pd.DataFrame(all_rows)
        out_df.to_csv(out_path, sep="\t", index=False)
        print(f"Wrote relaxed window betas ({design}) to: {out_path}")
        print(f"Total rows ({design}): {len(out_df)}")

    print("=== Model_1B_relaxed(v3): done ===")


if __name__ == "__main__":
    main()