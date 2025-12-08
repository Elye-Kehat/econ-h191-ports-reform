from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Toggle these to include SIPG / Ashdod K/L specs once the data exist.
INCLUDE_SIPG_KL: bool = False
INCLUDE_ASHDOD_KL: bool = False

# Event-time window for dynamic plots / windows
MIN_EVENT_TIME: int = -12
MAX_EVENT_TIME: int = 24

# Names for model / output
MODEL_NAME: str = "1B"


# ---------------------------------------------------------------------------
# Helpers for paths
# ---------------------------------------------------------------------------

def find_thesis_root() -> Path:
    """
    Heuristic: this file is in THESIS/Design/Code (new)/.
    Thesis root is two parents up from this file.
    """
    here = Path(__file__).resolve()
    # ... /Thesis/Design/Code (new)/Model_1B.py
    return here.parents[2]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BaseSpecKL:
    """
    Base definition of an event-study spec for K/L.

    * reform: short code, e.g. 'haifa_comp', 'haifa_priv'.
    * target: human-readable label, e.g. 'Haifa-Legacy terminal'.
    * event_year, event_month: calendar date of the reform (time 0).
    * treated_labels: labels of series that are truly treated.
    * control_labels: labels of “never-treated” controls for NYT design.
    """
    reform: str
    target: str
    event_year: int
    event_month: int
    treated_labels: List[str]
    control_labels: List[str]


@dataclass
class SpecFEKL:
    """
    Fully specified spec including FE / controls variant.
    """
    base: BaseSpecKL
    fe_type: str  # 'baseline', 'porttr', 'tr_shocks'


# ---------------------------------------------------------------------------
# Series label → (level, port, terminal) filters
# ---------------------------------------------------------------------------

LABEL_FILTERS_KL: Dict[str, Dict[str, str]] = {
    # Haifa
    "Haifa port":    {"level": "port",     "port": "Haifa"},
    "Haifa-Legacy":  {"level": "terminal", "port": "Haifa", "terminal": "Haifa-Legacy"},
    "Haifa-Bayport": {"level": "terminal", "port": "Haifa", "terminal": "Haifa-Bayport"},

    # Ashdod – placeholders for future series
    "Ashdod port":    {"level": "port",     "port": "Ashdod"},
    "Ashdod-Legacy":  {"level": "terminal", "port": "Ashdod", "terminal": "Ashdod-Legacy"},
    "Ashdod-HCT":     {"level": "terminal", "port": "Ashdod", "terminal": "Ashdod-HCT"},
}


def safe_get_series_id_kl(df: pd.DataFrame, label: str) -> Optional[str]:
    """
    Map a human-readable label (e.g. 'Haifa-Legacy') to a unique series_id in
    the K/L panel, using LABEL_FILTERS_KL.

    Returns:
        series_id (str) if a unique match is found,
        None otherwise (with a warning printed).
    """
    if label not in LABEL_FILTERS_KL:
        print(f"[WARN] LABEL_FILTERS_KL has no entry for label {label!r}.")
        return None

    flt = LABEL_FILTERS_KL[label]
    mask = np.ones(len(df), dtype=bool)
    for col, val in flt.items():
        if col not in df.columns:
            print(f"[WARN] Column {col!r} not in KL panel when matching label {label!r}.")
            return None
        mask &= (df[col] == val)

    candidates = df.loc[mask, "series_id"].dropna().unique()
    if len(candidates) == 0:
        uniq = df[["series_id", "level", "freq", "port", "terminal"]].drop_duplicates().head(5)
        print(
            f"[WARN] In KL panel, no unique series_id matched label {label!r}. "
            f"Skipping this label. Here are some example series:\n{uniq.to_string(index=False)}"
        )
        return None
    if len(candidates) > 1:
        print(
            f"[WARN] Multiple series_ids {candidates!r} matched label {label!r}. "
            "Using the first one; check that this is what you want."
        )

    return str(candidates[0])


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------

def build_base_specs_kl() -> List[BaseSpecKL]:
    """
    Build the list of base K/L specs that Model 1B will estimate.

    For now, with only Haifa K/L data, we focus on Haifa-Legacy as the
    treated terminal, with Haifa port-cluster as the never-treated control.
    """
    specs: List[BaseSpecKL] = []

    # 1) Haifa competition entry – effect on Haifa-Legacy K/L
    specs.append(
        BaseSpecKL(
            reform="haifa_comp",
            target="Haifa-Legacy terminal",
            event_year=2021,
            event_month=9,   # SIPG Bayport effective 09-2021
            treated_labels=["Haifa-Legacy"],
            control_labels=["Haifa port"],
        )
    )

    # 2) Haifa privatization – effect on Haifa-Legacy K/L
    specs.append(
        BaseSpecKL(
            reform="haifa_priv",
            target="Haifa-Legacy terminal",
            event_year=2023,
            event_month=1,   # Haifa privatization effective 01-2023
            treated_labels=["Haifa-Legacy"],
            control_labels=["Haifa port"],
        )
    )

    # Future Ashdod / SIPG specs (stay off until K/L exists)
    if INCLUDE_ASHDOD_KL:
        specs.append(
            BaseSpecKL(
                reform="ashdod_comp",
                target="Ashdod-HCT terminal",
                event_year=2022,
                event_month=11,
                treated_labels=["Ashdod-HCT"],
                control_labels=["Haifa port", "Haifa-Legacy"],
            )
        )

    if INCLUDE_SIPG_KL:
        specs.append(
            BaseSpecKL(
                reform="haifa_comp",
                target="Haifa-Bayport terminal",
                event_year=2021,
                event_month=9,
                treated_labels=["Haifa-Bayport"],
                control_labels=["Ashdod-HCT", "Ashdod-Legacy", "Ashdod port"],
            )
        )

    return specs


def expand_specs_with_fe(base_specs: List[BaseSpecKL]) -> List[SpecFEKL]:
    """
    Attach FE variants to each base spec: baseline, +port trends, +trends&shocks.
    """
    fe_types = ["baseline", "porttr", "tr_shocks"]
    specs_fe: List[SpecFEKL] = []
    for base in base_specs:
        for fe in fe_types:
            specs_fe.append(SpecFEKL(base=base, fe_type=fe))
    return specs_fe


# ---------------------------------------------------------------------------
# Build event-study sample for a given spec
# ---------------------------------------------------------------------------

def build_es_sample_kl(
    df_kl: pd.DataFrame,
    spec: BaseSpecKL,
    ym_to_idx: Dict[Tuple[int, int], int],
) -> Optional[pd.DataFrame]:
    """
    Construct a stacked panel for the NYT-style event study for a given
    base spec.

    * Treated units get event_time = month_index - event_month_index.
    * Control units are forced to event_time = -1 (reference period) in
      all months to deliver a not-yet-treated comparison.
    """
    event_key = (spec.event_year, spec.event_month)
    if event_key not in ym_to_idx:
        print(
            f"[WARN] Event date {event_key} is outside the KL panel support "
            f"for {spec.reform} – {spec.target}. Skipping this spec."
        )
        return None

    event_idx = ym_to_idx[event_key]

    # Map labels → series_ids
    treated_ids: List[str] = []
    control_ids: List[str] = []

    for lbl in spec.treated_labels:
        sid = safe_get_series_id_kl(df_kl, lbl)
        if sid is not None:
            treated_ids.append(sid)

    for lbl in spec.control_labels:
        sid = safe_get_series_id_kl(df_kl, lbl)
        if sid is not None:
            control_ids.append(sid)

    treated_ids = sorted(set(treated_ids))
    control_ids = sorted(set(control_ids))

    if len(treated_ids) == 0:
        print(
            f"[WARN] For {spec.reform} – {spec.target}, no treated series_ids "
            "found in KL panel; skipping."
        )
        return None
    if len(control_ids) == 0:
        print(
            f"[WARN] For {spec.reform} – {spec.target}, no control series_ids "
            "found in KL panel; skipping."
        )
        return None

    # Build sample from relevant series
    df = df_kl[df_kl["series_id"].isin(treated_ids + control_ids)].copy()

    # Basic indicators
    df["treat"] = df["series_id"].isin(treated_ids).astype(int)
    df["event_idx"] = event_idx
    df["event_time"] = df["month_index"] - df["event_idx"]

    # Impose event window (plus the reference period -1)
    lo = MIN_EVENT_TIME - 1  # ensure -1 is inside range
    hi = MAX_EVENT_TIME
    df = df[(df["event_time"] >= lo) & (df["event_time"] <= hi)].copy()

    # NYT-style event_time for treated vs controls:
    #  * Treated units keep their true event_time.
    #  * Control units are always in the reference bin (-1).
    df["event_time_treat"] = np.where(df["treat"] == 1, df["event_time"], -1)

    # Book-keeping counts for viability
    treated_n = df.loc[df["treat"] == 1, "log_KL"].notna().sum()
    control_n = df.loc[df["treat"] == 0, "log_KL"].notna().sum()

    if treated_n == 0 or control_n == 0:
        print(
            f"[WARN] For {spec.reform} – {spec.target}, treated_n={treated_n}, "
            f"control_n={control_n}. Need both treated and control observations; skipping."
        )
        return None

    return df


# ---------------------------------------------------------------------------
# Regression + extraction of dynamic and window betas
# ---------------------------------------------------------------------------

def run_es_regression_kl(
    df_es: pd.DataFrame,
    spec_fe: SpecFEKL,
    shock_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run one ES regression and return three small DataFrames:

        dynamic_betas_rows, window_betas_rows, pretrend_tests_rows
    """
    base = spec_fe.base
    fe_type = spec_fe.fe_type

    # Ensure we have a cluster_id (for clustering by port where possible)
    if "cluster_id" not in df_es.columns:
        if "port" in df_es.columns:
            df_es = df_es.copy()
            df_es["cluster_id"] = df_es["port"].astype(str)
        else:
            df_es = df_es.copy()
            df_es["cluster_id"] = df_es["series_id"].astype(str)

    cluster_by = "cluster_id"

    # Build RHS formula terms
    rhs_terms: List[str] = []

    # Event-time dummies (treated vs ref period)
    rhs_terms.append("C(event_time_treat, Treatment(reference=-1))")

    # Terminal FE (here: series_id FE)
    rhs_terms.append("C(series_id)")

    # Calendar-month FE (global time index)
    rhs_terms.append("C(month_index)")

    # Port-specific linear trends if requested
    if fe_type in ("porttr", "tr_shocks"):
        df_es = df_es.copy()
        t0 = df_es["month_index"].min()
        df_es["t_index"] = df_es["month_index"] - t0

        trend_cols: List[str] = []
        for port_val in sorted(df_es["port"].dropna().unique()):
            col = f"trend_{port_val}"
            df_es[col] = np.where(df_es["port"] == port_val, df_es["t_index"], 0.0)
            trend_cols.append(col)

        rhs_terms.extend(trend_cols)

    # Shock controls if requested and available
    if fe_type == "tr_shocks":
        for col in shock_cols:
            if col in df_es.columns:
                rhs_terms.append(col)

    formula = "log_KL ~ " + " + ".join(rhs_terms)
    print(f"  OLS formula: {formula}")

    model = smf.ols(formula, data=df_es)

    # Covariance handling:
    # - If we have at least 2 clusters, prefer cluster-robust SEs clustered by port.
    # - If there is only 1 cluster, or cluster-robust fails, fall back to HC1.
    cluster_series = df_es[cluster_by]
    n_clusters = cluster_series.nunique()

    if n_clusters < 2:
        print(
            f"[WARN] Only {n_clusters} cluster(s) in this ES sample; "
            "using heteroskedastic-robust (HC1) covariance for this spec."
        )
        result = model.fit(cov_type="HC1")
    else:
        base_res = model.fit()
        try:
            result = base_res.get_robustcov_results(
                cov_type="cluster",
                groups=cluster_series,
            )
        except ZeroDivisionError:
            print(
                "[WARN] Cluster-robust covariance failed (likely nobs - k_params == 0); "
                "falling back to heteroskedastic-robust (HC1) covariance for this spec."
            )
            result = base_res.get_robustcov_results(cov_type="HC1")

    # ------------------------------------------------------------------
    # Dynamic betas
    # ------------------------------------------------------------------
    params = result.params
    bse = result.bse
    tvals = result.tvalues
    pvals = result.pvalues

    dyn_rows: List[Dict] = []

    treated_n = int(df_es.loc[df_es["treat"] == 1, "log_KL"].notna().sum())
    control_n = int(df_es.loc[df_es["treat"] == 0, "log_KL"].notna().sum())

    # spec_name will just be the FE type; reform/target identify the rest
    spec_label = fe_type

    for m in range(MIN_EVENT_TIME, MAX_EVENT_TIME + 1):
        if m == -1:
            # reference bin – no coefficient
            continue
        term = f"C(event_time_treat, Treatment(reference=-1))[T.{m}]"
        beta = params.get(term, np.nan)
        se = bse.get(term, np.nan)
        tval = tvals.get(term, np.nan)
        pval = pvals.get(term, np.nan)

        dyn_rows.append(
            dict(
                model=MODEL_NAME,
                reform=base.reform,
                target=base.target,
                spec_name=spec_label,
                event_time=m,
                beta_hat=beta,
                se=se,
                tvalue=tval,
                pvalue=pval,
                n_obs=int(result.nobs),
                treated_n=treated_n,
                control_n=control_n,
                fe_type=fe_type,
                cov_type=result.cov_type,
                cluster_by=cluster_by,
                r2=result.rsquared,
            )
        )

    dynamic_df = pd.DataFrame(dyn_rows)

    # ------------------------------------------------------------------
    # Window betas
    # ------------------------------------------------------------------
    cov = result.cov_params()

    def window_stats(name: str, m_list: List[int]) -> Dict:
        terms = [
            f"C(event_time_treat, Treatment(reference=-1))[T.{m}]"
            for m in m_list
            if f"C(event_time_treat, Treatment(reference=-1))[T.{m}]" in params.index
        ]
        if not terms:
            return dict(
                model=MODEL_NAME,
                reform=base.reform,
                target=base.target,
                spec_name=spec_label,
                window_name=name,
                m_start=min(m_list),
                m_end=max(m_list),
                beta_hat=np.nan,
                se=np.nan,
                tvalue=np.nan,
                pvalue=np.nan,
                n_obs=int(result.nobs),
                treated_n=treated_n,
                control_n=control_n,
                fe_type=fe_type,
                cov_type=result.cov_type,
                cluster_by=cluster_by,
                r2=result.rsquared,
            )

        idx = [params.index.get_loc(t) for t in terms]
        b = params.iloc[idx].values
        # Equal weights over the included m's
        w = np.ones(len(terms)) / len(terms)
        # Robust variance for w' * beta
        cov_sub = cov.to_numpy()[np.ix_(idx, idx)]
        var = float(w @ cov_sub @ w)
        se = np.sqrt(var) if var >= 0 else np.nan
        beta_hat = float(w @ b)
        tvalue = beta_hat / se if se > 0 else np.nan

        # Approximate two-sided p-value using normal approximation
        if np.isnan(tvalue):
            pvalue = np.nan
        else:
            pvalue = 2 * (1 - 0.5 * (1 + np.math.erf(abs(tvalue) / np.sqrt(2))))

        return dict(
            model=MODEL_NAME,
            reform=base.reform,
            target=base.target,
            spec_name=spec_label,
            window_name=name,
            m_start=min(m_list),
            m_end=max(m_list),
            beta_hat=beta_hat,
            se=se,
            tvalue=tvalue,
            pvalue=pvalue,
            n_obs=int(result.nobs),
            treated_n=treated_n,
            control_n=control_n,
            fe_type=fe_type,
            cov_type=result.cov_type,
            cluster_by=cluster_by,
            r2=result.rsquared,
        )

    win_rows: List[Dict] = []
    # All post-reform months (1..MAX_EVENT_TIME)
    win_rows.append(window_stats("post_all", list(range(1, MAX_EVENT_TIME + 1))))
    # Year 1 post (1..12)
    win_rows.append(window_stats("post_y1", list(range(1, min(12, MAX_EVENT_TIME) + 1))))
    # Year 2 post (13..24) if available
    if MAX_EVENT_TIME >= 13:
        win_rows.append(window_stats("post_y2", list(range(13, min(24, MAX_EVENT_TIME) + 1))))
    # Pre-trend window (MIN_EVENT_TIME..-2)
    pre_ms = [m for m in range(MIN_EVENT_TIME, 0) if m != -1]
    win_rows.append(window_stats("pre_all", pre_ms))

    window_df = pd.DataFrame(win_rows)

    # ------------------------------------------------------------------
    # Pre-trend F-test: H0: all β_m = 0 for m <= -2
    # Use NON-ROBUST OLS for this test (like Model 1A), to avoid
    # the statsmodels restriction on F-tests with robust covariance.
    # ------------------------------------------------------------------
    pre_terms = [
        f"C(event_time_treat, Treatment(reference=-1))[T.{m}]"
        for m in range(MIN_EVENT_TIME, 0)
        if m != -1 and f"C(event_time_treat, Treatment(reference=-1))[T.{m}]" in params.index
    ]
    if pre_terms:
        R = np.zeros((len(pre_terms), len(params)))
        param_index = {name: j for j, name in enumerate(params.index)}
        for i, term in enumerate(pre_terms):
            R[i, param_index[term]] = 1.0

        # Plain OLS (non-robust) for pre-trend F-test
        try:
            ols_res = smf.ols(formula, data=df_es).fit()

            df_resid = float(ols_res.df_resid)
            if df_resid <= 0:
                print(
                    f"[WARN] Pre-trend F-test skipped for {base.reform} – {base.target} "
                    f"[{fe_type}]: df_resid={df_resid:.1f} ≤ 0; model is saturated."
                )
                stat = np.nan
                pval = np.nan
                df_denom = df_resid
                df_num = float(len(pre_terms))
            else:
                ftest = ols_res.f_test(R)
                stat = float(ftest.fvalue)
                pval = float(ftest.pvalue)
                df_denom = df_resid
                df_num = float(len(pre_terms))

        except Exception as e:
            print(
                f"[WARN] Pre-trend F-test failed for {base.reform} – {base.target} "
                f"[{fe_type}]: {e}"
            )
            stat = np.nan
            pval = np.nan
            df_denom = np.nan
            df_num = np.nan
    else:
        stat = np.nan
        pval = np.nan
        df_denom = np.nan
        df_num = np.nan

    pretrend_row = dict(
        model=MODEL_NAME,
        reform=base.reform,
        target=base.target,
        spec_name=spec_label,
        test_name="pretrend_all_m_le_-2",
        stat=stat,
        df_num=df_num,
        df_denom=df_denom,
        pvalue=pval,
        fe_type=fe_type,
        cov_type=result.cov_type,  # covariance used for main betas
        cluster_by=cluster_by,
        r2=result.rsquared,
    )

    pretrend_df = pd.DataFrame([pretrend_row])

    return dynamic_df, window_df, pretrend_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Model_1B (K/L) event-study: starting ===")

    thesis_root = find_thesis_root()
    print(f"Thesis root: {thesis_root}")

    kl_panel_path = thesis_root / "Data" / "KL" / "KL_Panel_monthly.tsv"
    out_dir = thesis_root / "Design" / "Output (new)" / "Model_1B"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"KL panel path: {kl_panel_path}")
    print(f"Output directory: {out_dir}")

    # ------------------------------------------------------------------
    # Read KL panel
    # ------------------------------------------------------------------
    df_kl = pd.read_csv(kl_panel_path, sep="\t")
    print(f"Loaded {len(df_kl)} rows from KL_Panel_monthly.tsv.")

    # Ensure log_KL exists
    if "log_KL" not in df_kl.columns:
        if "KL" not in df_kl.columns:
            raise ValueError(
                "KL_Panel_monthly.tsv must contain 'KL' or 'log_KL'. "
                "Neither column is present."
            )
        df_kl = df_kl.copy()
        df_kl["log_KL"] = np.log(df_kl["KL"])

    # Build (year, month) → sequential month_index
    df_kl = df_kl.copy()
    df_kl["ym_tuple"] = list(zip(df_kl["year"], df_kl["month"]))
    ym_sorted = sorted(df_kl["ym_tuple"].unique())
    ym_to_idx: Dict[Tuple[int, int], int] = {ym: i + 1 for i, ym in enumerate(ym_sorted)}
    df_kl["month_index"] = df_kl["ym_tuple"].map(ym_to_idx)

    print(f"Built (year, month) -> month_index mapping for {len(ym_sorted)} months.")

    # Shock controls present in the KL panel
    shock_cols = [c for c in ["covid_shock", "war_shock"] if c in df_kl.columns]
    if shock_cols:
        print(f"Shock control columns detected in KL panel: {shock_cols}")
    else:
        print("No shock control columns detected in KL panel.")

    # Pre-compute cluster_id at the panel level (port-based where possible)
    if "port" in df_kl.columns:
        df_kl["cluster_id"] = df_kl["port"].astype(str)
    else:
        df_kl["cluster_id"] = df_kl["series_id"].astype(str)

    # ------------------------------------------------------------------
    # Build specs
    # ------------------------------------------------------------------
    base_specs = build_base_specs_kl()
    specs_fe = expand_specs_with_fe(base_specs)
    print(f"Total Spec×FE combinations for Model 1B: {len(specs_fe)}")

    all_dynamic_rows: List[pd.DataFrame] = []
    all_window_rows: List[pd.DataFrame] = []
    all_pretrend_rows: List[pd.DataFrame] = []

    for spec_fe in specs_fe:
        base = spec_fe.base
        label = f"{base.reform} – {base.target} [{spec_fe.fe_type}]"
        print(f"\n--- Processing spec: {label} ---")

        df_es = build_es_sample_kl(df_kl, base, ym_to_idx)
        if df_es is None:
            continue

        dynamic_df, window_df, pretrend_df = run_es_regression_kl(df_es, spec_fe, shock_cols)

        all_dynamic_rows.append(dynamic_df)
        all_window_rows.append(window_df)
        all_pretrend_rows.append(pretrend_df)

    if not all_dynamic_rows:
        print(
            "[WARN] No successful specs for Model 1B (K/L). "
            "This likely reflects incomplete K/L coverage (e.g., "
            "no suitable treated/control series in the KL panel for the current specs)."
        )
        print("=== Model_1B: done (no outputs written) ===")
        return

    # Concatenate and write outputs
    dyn_all = pd.concat(all_dynamic_rows, ignore_index=True)
    win_all = pd.concat(all_window_rows, ignore_index=True)
    pre_all = pd.concat(all_pretrend_rows, ignore_index=True)

    dyn_path = out_dir / "model1b_kl_dynamic_betas_all.tsv"
    win_path = out_dir / "model1b_kl_window_betas_all.tsv"
    pre_path = out_dir / "model1b_kl_pretrend_tests_all.tsv"

    dyn_all.to_csv(dyn_path, sep="\t", index=False)
    win_all.to_csv(win_path, sep="\t", index=False)
    pre_all.to_csv(pre_path, sep="\t", index=False)

    print(f"\nWrote dynamic betas to:   {dyn_path}")
    print(f"Wrote window betas to:    {win_path}")
    print(f"Wrote pretrend tests to:  {pre_path}")
    print("=== Model_1B: done ===")


if __name__ == "__main__":
    main()
