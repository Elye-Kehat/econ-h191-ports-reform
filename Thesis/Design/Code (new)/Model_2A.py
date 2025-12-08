
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Root of the thesis repository (adjust parents[...] if your folder depth differs)
THESIS_ROOT = Path(__file__).resolve().parents[2]

# Input data paths
LP_PANEL_MONTHLY_PATH = THESIS_ROOT / "Data" / "LP" / "LP_Panel_monthly.tsv"
KL_PANEL_MONTHLY_PATH = THESIS_ROOT / "Data" / "KL" / "KL_Panel_monthly.tsv"

# Output directory
OUT_DIR = THESIS_ROOT / "Design" / "Output (new)" / "Model_2A"

# Series identifiers (must match LP_Panel_monthly.tsv and KL_Panel_monthly.tsv)
LP_SERIES_ID = "Haifa_Legacy_Q"       # terminal-level LP on a monthly grid
KL_SERIES_ID = "Haifa_Legacy_KL"      # terminal-level K/L

# HAC lags for monthly data
HAC_LAGS_M = 6


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def ensure_outdir(path: Path) -> None:
    """Create output directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def build_monthly_panel() -> pd.DataFrame:
    """
    Build the monthly Haifa-Legacy terminal panel for Model 2A.

    - Reads LP from LP_Panel_monthly.tsv (series_id == Haifa_Legacy_Q).
    - Reads K/L from KL_Panel_monthly.tsv (series_id == Haifa_Legacy_KL).
    - Merges them on (year, month).
    - Constructs a monthly time index t_index.
    """
    # Load monthly LP for Haifa Legacy (quarter-coded but on a monthly grid)
    lp = pd.read_csv(LP_PANEL_MONTHLY_PATH, sep="\t")
    lp = lp.loc[lp["series_id"] == LP_SERIES_ID].copy()

    # Sort by calendar time
    lp = lp.sort_values(["year", "month"])

    # Construct log(LP) and drop rows with missing LP
    lp["log_LP"] = np.log(lp["LP"])
    lp = lp.loc[lp["log_LP"].notna()].copy()

    # Load monthly K/L for Haifa Legacy
    kl_m = pd.read_csv(KL_PANEL_MONTHLY_PATH, sep="\t")
    kl_m = kl_m.loc[kl_m["series_id"] == KL_SERIES_ID].copy()
    kl_m = kl_m.sort_values(["year", "month"])

    # Keep only necessary columns
    kl_m_slim = kl_m[["year", "month", "KL", "log_KL"]].copy()

    # Monthly merge: LP (Haifa_Legacy_Q) x KL (Haifa_Legacy_KL)
    df = lp.merge(kl_m_slim, on=["year", "month"], how="inner")

    # Sort and build monthly time index
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df["t_index"] = np.arange(len(df))

    return df


def run_regressions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Model 2A regressions: log(LP_t) on log(K/L)_t (and trend).

    Specifications:
    - baseline: log_LP ~ const + log_KL
    - trend:    log_LP ~ const + log_KL + t_index

    Uses HAC (Newey–West) standard errors with monthly lags.
    Returns a tidy DataFrame of results.
    """
    specs = {
        "baseline": ["log_KL"],
        "trend": ["log_KL", "t_index"],
    }

    results_rows = []

    for spec_name, rhs_vars in specs.items():
        X = df[rhs_vars].copy()
        X = sm.add_constant(X)

        y = df["log_LP"].copy()

        model = sm.OLS(y, X)
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS_M})

        n_obs = int(res.nobs)
        r2 = float(res.rsquared)
        r2_adj = float(res.rsquared_adj)

        # Extract statistics for log_KL
        if "log_KL" in res.params.index:
            coef_log_KL = float(res.params["log_KL"])
            se_log_KL = float(res.bse["log_KL"])
            t_log_KL = float(res.tvalues["log_KL"])
            p_log_KL = float(res.pvalues["log_KL"])
        else:
            coef_log_KL = np.nan
            se_log_KL = np.nan
            t_log_KL = np.nan
            p_log_KL = np.nan

        results_rows.append(
            {
                "spec_name": spec_name,
                "coef_log_KL": coef_log_KL,
                "se_log_KL": se_log_KL,
                "t_log_KL": t_log_KL,
                "p_log_KL": p_log_KL,
                "n_obs": n_obs,
                "r2": r2,
                "r2_adj": r2_adj,
            }
        )

        print(f"[Model 2A] Finished spec '{spec_name}': n_obs={n_obs}, coef_log_KL={coef_log_KL:.3f}")

    return pd.DataFrame(results_rows)


def main() -> None:
    ensure_outdir(OUT_DIR)

    # Build monthly terminal panel and write it out
    df_panel = build_monthly_panel()
    panel_path = OUT_DIR / "model2a_terminal_panel.tsv"
    df_panel.to_csv(panel_path, sep="\t", index=False)
    print(f"[Model 2A] Wrote monthly terminal panel to: {panel_path} (n={len(df_panel)})")

    # Run regressions and write results
    df_results = run_regressions(df_panel)
    results_path = OUT_DIR / "model2a_reg_results.tsv"
    df_results.to_csv(results_path, sep="\t", index=False)
    print(f"[Model 2A] Wrote regression results to: {results_path}")


if __name__ == "__main__":
    main()
