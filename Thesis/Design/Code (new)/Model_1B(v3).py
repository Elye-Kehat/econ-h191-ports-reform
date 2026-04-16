from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# =============================================================================
# Model_1B(v3): K/L event-study with NYT + TWFE designs
#
# v3 patches:
#  (A) NYT saturation guard:
#      For NYT only, automatically shrink max post horizon if the implied
#      parameter count would drive df_resid near zero (causing NaN SEs).
#      This is NOT a hard cap at 12; it chooses the largest max_post that
#      leaves a minimum df_resid buffer.
#
#  (B) Consistency / minimal clutter:
#      - NYT outputs keep legacy filenames (no suffix).
#      - TWFE outputs are pooled-only with suffix "_twfe".
#      - Add j = event_time in outputs.
#      - Add "did_post" window row (treated × post>=1 TWFE) inside window TSVs.
# =============================================================================


# --------------------------
# Global settings
# --------------------------

MIN_EVENT_TIME: int = -12
MAX_EVENT_TIME: int = 24

# Minimum estimated residual degrees of freedom we try to preserve in NYT
# (purely a numerical guard against saturation in small stacked samples)
NYT_MIN_DF_RESID_TARGET: int = 8

MODEL_NAME: str = "1B"


# --------------------------
# Paths
# --------------------------

def find_thesis_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2]


# --------------------------
# Spec structures
# --------------------------

@dataclass
class BaseSpecKL:
    reform: str
    target: str
    event_year: int
    event_month: int
    treated_labels: List[str]
    control_labels: List[str]


@dataclass
class SpecFEKL:
    base: BaseSpecKL
    fe_type: str  # baseline / porttr / tr_shocks


# --------------------------
# Label mapping
# --------------------------

LABEL_FILTERS_KL: Dict[str, Dict[str, str]] = {
    # Haifa terminals
    "Haifa-Legacy":  {"level": "terminal", "port": "Haifa", "terminal": "Haifa-Legacy"},
    "Haifa-Bayport": {"level": "terminal", "port": "Haifa", "terminal": "Haifa-Bayport"},

    # K/L constructed cluster series (expected to exist by series_id)
    "Haifa port cluster (central)": {"series_id": "Haifa_port_KL_cluster_central"},
    "Haifa-Bayport K/L (SIPG central)": {"series_id": "Haifa_port_KL_SIPG_central"},

    # Ashdod placeholders for future
    "Ashdod-Legacy": {"level": "terminal", "port": "Ashdod", "terminal": "Ashdod-Legacy"},
    "Ashdod-HCT":    {"level": "terminal", "port": "Ashdod", "terminal": "Ashdod-HCT"},
}


def safe_get_series_id_kl(df: pd.DataFrame, label: str) -> Optional[str]:
    if label not in LABEL_FILTERS_KL:
        print(f"[WARN] LABEL_FILTERS_KL missing label {label!r}")
        return None

    flt = LABEL_FILTERS_KL[label]
    mask = np.ones(len(df), dtype=bool)
    for col, val in flt.items():
        if col not in df.columns:
            print(f"[WARN] Column {col!r} not found in KL panel; cannot match {label!r}")
            return None
        mask &= (df[col] == val)

    sids = df.loc[mask, "series_id"].dropna().unique()
    if len(sids) == 0:
        print(f"[WARN] No series_id matched label {label!r}")
        return None
    if len(sids) > 1:
        print(f"[WARN] Multiple series_ids matched label {label!r}: {sids}. Using first.")
    return str(sids[0])


# --------------------------
# Base specs
# --------------------------

def build_base_specs_kl() -> List[BaseSpecKL]:
    specs: List[BaseSpecKL] = []

    # Haifa competition (2021-09)
    specs.append(
        BaseSpecKL(
            reform="haifa_comp",
            target="Haifa-Legacy K/L",
            event_year=2021,
            event_month=9,
            treated_labels=["Haifa-Legacy"],
            control_labels=["Haifa port cluster (central)"],
        )
    )

    # Haifa privatization (2023-01)
    specs.append(
        BaseSpecKL(
            reform="haifa_priv",
            target="Haifa-Legacy K/L",
            event_year=2023,
            event_month=1,
            treated_labels=["Haifa-Legacy"],
            control_labels=["Haifa port cluster (central)"],
        )
    )

    # Bayport/SIPG central series (if present)
    specs.append(
        BaseSpecKL(
            reform="haifa_comp",
            target="Haifa-Bayport K/L (SIPG central)",
            event_year=2021,
            event_month=9,
            treated_labels=["Haifa-Bayport K/L (SIPG central)"],
            control_labels=["Haifa-Legacy"],
        )
    )

    specs.append(
        BaseSpecKL(
            reform="haifa_priv",
            target="Haifa-Bayport K/L (SIPG central)",
            event_year=2023,
            event_month=1,
            treated_labels=["Haifa-Bayport K/L (SIPG central)"],
            control_labels=["Haifa-Legacy"],
        )
    )

    return specs


def expand_specs_with_fe(base_specs: List[BaseSpecKL], shock_cols: List[str]) -> List[SpecFEKL]:
    fe_types = ["baseline", "porttr"]
    if shock_cols:
        fe_types.append("tr_shocks")
    return [SpecFEKL(base=b, fe_type=fe) for b in base_specs for fe in fe_types]


# --------------------------
# NYT saturation guard: choose max_post for NYT if needed
# --------------------------

def _estimate_df_resid(nobs: int, n_series: int, n_months: int, n_event_dummies: int) -> int:
    # Intercept + (series FE) + (month FE) + (event dummies)
    # series FE contributes (n_series-1), month FE contributes (n_months-1)
    k = 1 + (n_series - 1) + (n_months - 1) + n_event_dummies
    return int(nobs - k)


def choose_nyt_max_post(df_raw: pd.DataFrame, base_max: int) -> int:
    """
    Choose the largest max_post <= base_max such that an approximate df_resid
    stays above NYT_MIN_DF_RESID_TARGET.

    df_raw should already include treat/control series and have event_time computed.
    """
    max_post_data = int(df_raw["event_time"].max())
    hi0 = min(base_max, max_post_data)

    # Search from hi0 downward until df_resid target met
    for hi in range(hi0, 0, -1):
        df = df_raw[(df_raw["event_time"] >= (MIN_EVENT_TIME - 1)) & (df_raw["event_time"] <= hi)]
        if df.empty:
            continue

        nobs = len(df)
        n_series = int(df["series_id"].nunique())
        n_months = int(df["month_index"].nunique())

        treated = df[df["treat"] == 1]
        treated_ms = sorted(set(int(x) for x in treated["event_time"].unique()))
        treated_ms = [m for m in treated_ms if (MIN_EVENT_TIME <= m <= hi and m != -1)]
        n_event = len(treated_ms)

        df_resid_est = _estimate_df_resid(nobs, n_series, n_months, n_event)

        if df_resid_est >= NYT_MIN_DF_RESID_TARGET:
            return hi

    return hi0


# --------------------------
# Sample builder
# --------------------------

def build_es_sample_kl(
    df_kl: pd.DataFrame,
    spec: BaseSpecKL,
    ym_to_idx: Dict[Tuple[int, int], int],
    design: str = "nyt",
) -> Optional[pd.DataFrame]:
    if design not in {"nyt", "twfe"}:
        raise ValueError(f"design must be 'nyt' or 'twfe', got {design!r}")

    event_key = (spec.event_year, spec.event_month)
    if event_key not in ym_to_idx:
        print(f"[WARN] Event date {event_key} not in ym_to_idx for {spec.reform} – {spec.target}; skipping.")
        return None
    event_idx = ym_to_idx[event_key]

    treated_ids: List[str] = []
    for lbl in spec.treated_labels:
        sid = safe_get_series_id_kl(df_kl, lbl)
        if sid is not None:
            treated_ids.append(sid)
    treated_ids = sorted(set(treated_ids))
    if not treated_ids:
        print(f"[WARN] No treated series for {spec.reform} – {spec.target}; skipping.")
        return None

    if design == "nyt":
        control_ids: List[str] = []
        for lbl in spec.control_labels:
            sid = safe_get_series_id_kl(df_kl, lbl)
            if sid is not None:
                control_ids.append(sid)
        control_ids = sorted(set(control_ids))
        if not control_ids:
            print(f"[WARN] No control series for {spec.reform} – {spec.target} (NYT); skipping.")
            return None
    else:
        all_ids = sorted(df_kl["series_id"].dropna().unique().tolist())
        control_ids = [sid for sid in all_ids if sid not in treated_ids]
        control_ids = sorted(set(control_ids))
        if not control_ids:
            print(f"[WARN] No control series for {spec.reform} – {spec.target} (TWFE); skipping.")
            return None

    df = df_kl[df_kl["series_id"].isin(treated_ids + control_ids)].copy()
    df["treat"] = df["series_id"].isin(treated_ids).astype(int)

    # event time in months
    df["event_time"] = df["month_index"] - int(event_idx)
    df["j"] = df["event_time"]

    # Start with full horizon; NYT may shrink it below
    hi = MAX_EVENT_TIME

    # NYT saturation guard: pick max_post that leaves df_resid buffer
    if design == "nyt":
        hi_chosen = choose_nyt_max_post(df, base_max=MAX_EVENT_TIME)
        if hi_chosen < MAX_EVENT_TIME:
            print(f"  [NYT guard] Shrinking max_post from {MAX_EVENT_TIME} to {hi_chosen} for {spec.reform} – {spec.target}")
        hi = hi_chosen

    # Apply window (include MIN_EVENT_TIME-1 so j=-1 ref is kept when present)
    df = df[(df["event_time"] >= (MIN_EVENT_TIME - 1)) & (df["event_time"] <= hi)].copy()

    # Controls forced into reference bin for treated×event-time dummies
    df["event_time_treat"] = np.where(df["treat"] == 1, df["event_time"], -1)

    treated_n = int(df.loc[df["treat"] == 1, "log_KL"].notna().sum())
    control_n = int(df.loc[df["treat"] == 0, "log_KL"].notna().sum())
    if treated_n == 0 or control_n == 0:
        print(f"[WARN] {spec.reform} – {spec.target} [{design}]: treated_n={treated_n}, control_n={control_n}. Need both; skipping.")
        return None

    return df


# --------------------------
# Covariance helper
# --------------------------

def fit_with_covariance(model, df_es: pd.DataFrame, cluster_by: str) -> Tuple[object, str, str]:
    if cluster_by not in df_es.columns:
        return model.fit(cov_type="HC1"), "HC1", ""

    groups = df_es[cluster_by]
    n_clusters = int(groups.nunique())

    if n_clusters < 2:
        return model.fit(cov_type="HC1"), "HC1", ""

    base = model.fit()
    try:
        res = base.get_robustcov_results(cov_type="cluster", groups=groups)
        return res, "cluster", cluster_by
    except Exception:
        res = base.get_robustcov_results(cov_type="HC1")
        return res, "HC1", ""


def normal_p_two_sided(t: float) -> float:
    if not np.isfinite(t):
        return np.nan
    z = abs(float(t))
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return 2.0 * (1.0 - cdf)


# --------------------------
# Regression + extraction
# --------------------------

def run_es_regression_kl(df_es: pd.DataFrame, spec_fe: SpecFEKL, shock_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = spec_fe.base
    fe_type = spec_fe.fe_type
    df = df_es.copy()

    # cluster_id: port if possible else series_id
    if "cluster_id" not in df.columns:
        if "port" in df.columns:
            df["cluster_id"] = df["port"].astype(str)
        else:
            df["cluster_id"] = df["series_id"].astype(str)

    rhs_terms: List[str] = [
        "C(event_time_treat, Treatment(reference=-1))",
        "C(series_id)",
        "C(month_index)",
    ]

    # Port trends (applied as a single trend_Haifa etc.)
    if fe_type in ("porttr", "tr_shocks"):
        t0 = df["month_index"].min()
        df["t_index"] = df["month_index"] - t0
        if "port" in df.columns:
            for p in sorted(df["port"].dropna().unique()):
                col = f"trend_{p}"
                df[col] = np.where(df["port"] == p, df["t_index"], 0.0)
                rhs_terms.append(col)
        else:
            rhs_terms.append("t_index")

    if fe_type == "tr_shocks":
        for c in shock_cols:
            if c in df.columns:
                rhs_terms.append(c)

    formula = "log_KL ~ " + " + ".join(rhs_terms)
    print(f"  OLS formula: {formula}")

    model = smf.ols(formula, data=df)
    res, cov_type, cluster_used = fit_with_covariance(model, df, cluster_by="cluster_id")

    params = res.params
    bse = res.bse
    tvals = res.tvalues
    pvals = res.pvalues
    cov = res.cov_params()

    treated_n = int(df.loc[df["treat"] == 1, "log_KL"].notna().sum())
    control_n = int(df.loc[df["treat"] == 0, "log_KL"].notna().sum())

    # Dynamic β_j rows: keep full grid for stable plotting (NaN if unsupported)
    dyn_rows: List[Dict] = []
    for m in range(MIN_EVENT_TIME, MAX_EVENT_TIME + 1):
        if m == -1:
            continue
        term = f"C(event_time_treat, Treatment(reference=-1))[T.{m}]"
        dyn_rows.append(
            dict(
                model=MODEL_NAME,
                reform=base.reform,
                target=base.target,
                spec_name=fe_type,
                fe_type=fe_type,
                event_time=m,
                j=m,
                beta_hat=float(params.get(term, np.nan)),
                se=float(bse.get(term, np.nan)),
                tvalue=float(tvals.get(term, np.nan)),
                pvalue=float(pvals.get(term, np.nan)),
                n_obs=int(res.nobs),
                treated_n=treated_n,
                control_n=control_n,
                cov_type=cov_type,
                cluster_by=cluster_used,
                r2=float(res.rsquared),
            )
        )
    dynamic_df = pd.DataFrame(dyn_rows)

    # Identify available m’s (post support) from params
    available_ms: List[int] = []
    prefix = "C(event_time_treat, Treatment(reference=-1))[T."
    for nm in params.index:
        s = str(nm)
        if s.startswith(prefix) and s.endswith("]"):
            try:
                available_ms.append(int(s[len(prefix):-1]))
            except Exception:
                pass

    post_ms = sorted([m for m in available_ms if m >= 1])
    max_post_avail = max(post_ms) if post_ms else 0

    # Window averages via delta method on the linear combination
    def window_stats(name: str, ms: List[int]) -> Dict:
        terms = []
        used_ms = []
        for m in ms:
            term = f"{prefix}{m}]"
            if term in params.index:
                terms.append(term)
                used_ms.append(m)

        if not terms:
            return dict(
                model=MODEL_NAME, reform=base.reform, target=base.target,
                spec_name=fe_type, fe_type=fe_type,
                window_name=name, m_start=min(ms), m_end=max(ms),
                beta_hat=np.nan, se=np.nan, tvalue=np.nan, pvalue=np.nan,
                n_obs=int(res.nobs), treated_n=treated_n, control_n=control_n,
                cov_type=cov_type, cluster_by=cluster_used, r2=float(res.rsquared),
            )

        idx = [params.index.get_loc(t) for t in terms]
        b_vec = params.iloc[idx].values.astype(float)
        w = np.ones(len(idx), dtype=float) / len(idx)

        cov_sub = cov.to_numpy()[np.ix_(idx, idx)]
        var = float(w @ cov_sub @ w)
        se = float(np.sqrt(var)) if var >= 0 else np.nan

        beta_hat = float(w @ b_vec)
        tvalue = beta_hat / se if np.isfinite(se) and se > 0 else np.nan
        pvalue = normal_p_two_sided(tvalue) if np.isfinite(tvalue) else np.nan

        return dict(
            model=MODEL_NAME, reform=base.reform, target=base.target,
            spec_name=fe_type, fe_type=fe_type,
            window_name=name, m_start=int(min(used_ms)), m_end=int(max(used_ms)),
            beta_hat=beta_hat, se=se, tvalue=tvalue, pvalue=pvalue,
            n_obs=int(res.nobs), treated_n=treated_n, control_n=control_n,
            cov_type=cov_type, cluster_by=cluster_used, r2=float(res.rsquared),
        )

    win_rows: List[Dict] = []

    if max_post_avail >= 1:
        win_rows.append(window_stats("post_all", list(range(1, max_post_avail + 1))))
        win_rows.append(window_stats("post_y1", list(range(1, min(12, max_post_avail) + 1))))
        if max_post_avail >= 13:
            win_rows.append(window_stats("post_y2", list(range(13, max_post_avail + 1))))
        else:
            win_rows.append(window_stats("post_y2", list(range(13, 25))))
    else:
        win_rows.append(window_stats("post_all", list(range(1, 25))))
        win_rows.append(window_stats("post_y1", list(range(1, 13))))
        win_rows.append(window_stats("post_y2", list(range(13, 25))))

    pre_ms = [m for m in range(MIN_EVENT_TIME, 0) if m != -1]
    win_rows.append(window_stats("pre_all", pre_ms))

    # Static DiD row: treated × post_ge_1, with same FE structure
    df_did = df.copy()
    df_did["post_ge_1"] = (df_did["event_time"] >= 1).astype(int)
    df_did["treated_post"] = df_did["treat"] * df_did["post_ge_1"]

    rhs_did: List[str] = ["treated_post", "C(series_id)", "C(month_index)"]
    if fe_type in ("porttr", "tr_shocks"):
        if "t_index" not in df_did.columns:
            t0 = df_did["month_index"].min()
            df_did["t_index"] = df_did["month_index"] - t0
        if "port" in df_did.columns:
            for p in sorted(df_did["port"].dropna().unique()):
                col = f"trend_{p}"
                if col not in df_did.columns:
                    df_did[col] = np.where(df_did["port"] == p, df_did["t_index"], 0.0)
                rhs_did.append(col)
        else:
            rhs_did.append("t_index")
    if fe_type == "tr_shocks":
        for c in shock_cols:
            if c in df_did.columns:
                rhs_did.append(c)

    formula_did = "log_KL ~ " + " + ".join(rhs_did)
    model_did = smf.ols(formula_did, data=df_did)
    res_did, cov_type_did, cluster_used_did = fit_with_covariance(model_did, df_did, cluster_by="cluster_id")

    treated_post_js = df.loc[(df["treat"] == 1) & (df["event_time"] >= 1), "event_time"]
    max_post_eff = int(treated_post_js.max()) if len(treated_post_js) else 0

    win_rows.append(
        dict(
            model=MODEL_NAME, reform=base.reform, target=base.target,
            spec_name=fe_type, fe_type=fe_type,
            window_name="did_post", m_start=1, m_end=max_post_eff,
            beta_hat=float(res_did.params.get("treated_post", np.nan)),
            se=float(res_did.bse.get("treated_post", np.nan)),
            tvalue=float(res_did.tvalues.get("treated_post", np.nan)),
            pvalue=float(res_did.pvalues.get("treated_post", np.nan)),
            n_obs=int(res_did.nobs), treated_n=treated_n, control_n=control_n,
            cov_type=cov_type_did, cluster_by=cluster_used_did, r2=float(res_did.rsquared),
        )
    )

    window_df = pd.DataFrame(win_rows)

    # Pretrend test placeholder (keep structure; detailed test handled elsewhere)
    pretrend_df = pd.DataFrame(
        [{
            "model": MODEL_NAME,
            "reform": base.reform,
            "target": base.target,
            "spec_name": fe_type,
            "fe_type": fe_type,
            "test_name": "pretrend_all_m_le_-2",
            "stat": np.nan,
            "pvalue": np.nan,
            "df_num": np.nan,
            "df_denom": np.nan,
            "cov_type": cov_type,
            "cluster_by": cluster_used,
            "r2": float(res.rsquared),
        }]
    )

    return dynamic_df, window_df, pretrend_df


# --------------------------
# Run design
# --------------------------

def run_design(
    df_kl: pd.DataFrame,
    ym_to_idx: Dict[Tuple[int, int], int],
    shock_cols: List[str],
    design: str,
    suffix: str,
    out_dir: Path,
) -> None:
    base_specs = build_base_specs_kl()
    specs_fe = expand_specs_with_fe(base_specs, shock_cols=shock_cols)

    print(f"Total Spec×FE combinations for Model 1B [{design}]: {len(specs_fe)}")

    dyn_list: List[pd.DataFrame] = []
    win_list: List[pd.DataFrame] = []
    pre_list: List[pd.DataFrame] = []

    for spec_fe in specs_fe:
        base = spec_fe.base
        print(f"\n--- [{design}] Processing: {base.reform} – {base.target} [{spec_fe.fe_type}] ---")
        df_es = build_es_sample_kl(df_kl, base, ym_to_idx, design=design)
        if df_es is None or df_es.empty:
            continue

        n_treat = int(df_es["treat"].sum())
        print(f"  sample rows={len(df_es)} | treated_rows={n_treat} | controls_rows={len(df_es) - n_treat}")

        ddf, wdf, pdf = run_es_regression_kl(df_es, spec_fe, shock_cols)
        dyn_list.append(ddf)
        win_list.append(wdf)
        pre_list.append(pdf)

    if not dyn_list:
        print(f"[WARN] No results produced for design={design}")
        return

    dyn_all = pd.concat(dyn_list, ignore_index=True)
    win_all = pd.concat(win_list, ignore_index=True)
    pre_all = pd.concat(pre_list, ignore_index=True)

    dyn_path = out_dir / f"model1b_kl_dynamic_betas_all{suffix}.tsv"
    win_path = out_dir / f"model1b_kl_window_betas_all{suffix}.tsv"
    pre_path = out_dir / f"model1b_kl_pretrend_tests_all{suffix}.tsv"

    dyn_all.to_csv(dyn_path, sep="\t", index=False)
    win_all.to_csv(win_path, sep="\t", index=False)
    pre_all.to_csv(pre_path, sep="\t", index=False)

    print(f"\nWrote dynamic betas to:  {dyn_path}")
    print(f"Wrote window betas to:   {win_path}")
    print(f"Wrote pretrend tests to: {pre_path}")


# --------------------------
# Main
# --------------------------

def main() -> None:
    print("=== Model_1B(v3) (K/L) event-study: starting ===")
    thesis_root = find_thesis_root()
    print(f"Thesis root: {thesis_root}")

    kl_panel_path = thesis_root / "Data" / "KL" / "KL_Panel_monthly.tsv"
    out_dir = thesis_root / "Design" / "Output (new)" / "Model_1B"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"KL panel path: {kl_panel_path}")
    print(f"Output directory: {out_dir}")

    df_kl = pd.read_csv(kl_panel_path, sep="\t")
    print(f"Loaded {len(df_kl)} rows from KL_Panel_monthly.tsv.")

    # Ensure log_KL exists
    if "log_KL" not in df_kl.columns:
        if "KL" not in df_kl.columns:
            raise ValueError("KL_Panel_monthly.tsv must contain 'KL' or 'log_KL'.")
        df_kl = df_kl.copy()
        df_kl["log_KL"] = np.log(df_kl["KL"])

    # Build month_index mapping (overwrite to ensure consistency)
    df_kl = df_kl.copy()
    df_kl["ym_tuple"] = list(zip(df_kl["year"], df_kl["month"]))
    ym_sorted = sorted(df_kl["ym_tuple"].unique())
    ym_to_idx: Dict[Tuple[int, int], int] = {ym: i + 1 for i, ym in enumerate(ym_sorted)}
    df_kl["month_index"] = df_kl["ym_tuple"].map(ym_to_idx)
    print(f"Built (year, month) -> month_index mapping for {len(ym_sorted)} months.")

    shock_cols = [c for c in ["covid_shock", "war_shock"] if c in df_kl.columns]
    if shock_cols:
        print(f"Shock control columns detected in KL panel: {shock_cols}")
    else:
        print("No shock control columns detected in KL panel.")

    # cluster_id
    if "port" in df_kl.columns:
        df_kl["cluster_id"] = df_kl["port"].astype(str)
    else:
        df_kl["cluster_id"] = df_kl["series_id"].astype(str)

    print("\n==================== NYT run ====================")
    run_design(df_kl, ym_to_idx, shock_cols, design="nyt", suffix="", out_dir=out_dir)

    print("\n==================== TWFE run ====================")
    run_design(df_kl, ym_to_idx, shock_cols, design="twfe", suffix="_twfe", out_dir=out_dir)

    print("=== Model_1B(v3): done ===")


if __name__ == "__main__":
    main()




# =============================================================================
# MODEL_1B(v3) RUN EVALUATION (2026-03-xx)
#
# What worked:
# - v3 ran end-to-end and wrote both NYT (legacy filenames) and TWFE (_twfe) pooled outputs:
#     model1b_kl_{dynamic,window,pretrend}_all.tsv
#     model1b_kl_{dynamic,window,pretrend}_all_twfe.tsv
# - TWFE sample sizes expanded as intended (treated series + all other series_id as controls),
#   giving ~481–482 rows per spec and many more observations for month FE estimation.
# - Model_1B_to_tables(v3) successfully rebuilt N(m): Nm nonmissing = 288/288 for both NYT and TWFE,
#   eliminating the prior treated_n=0/control_n=0 skipping bug in the Nm builder.
# - Model_1B_relaxed(v3) ran NYT + TWFE and produced minimal outputs (2 files) with 16 rows/design.
#
# Important NYT inference note (expected):
# - NYT runs still trigger statsmodels "divide by zero" warnings. This reflects near-saturation in the
#   canonical NYT regression (tiny stacked sample with month FE + many event-time dummies), so NYT SEs
#   may be NaN/unreliable. This is why relaxed and Haifa TS variants exist; use those for inference.
# - The NYT guard occasionally shrinks max_post 24→23, but this is typically just aligning to data
#   support and does not by itself resolve saturation for NYT.
#
# Remaining fix (if needed for tables):
# - The pretrend-tests output in v3 is currently a placeholder (stat/pvalue NaN). If the paper needs
#   a "Pretrends F-test p-value" row for TWFE, restore an OLS f_test joint-leads test for TWFE
#   (NYT may remain undefined under saturation).
# =============================================================================