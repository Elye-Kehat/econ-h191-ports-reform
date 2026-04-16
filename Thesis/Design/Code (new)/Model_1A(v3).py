#!/usr/bin/env python
"""
Model_1A(v3).py

Event-study for log labor productivity ln(LP) using LP_Panel_monthly.tsv.

v3 goals:
  (1) Keep the NYT (not-yet-treated) design exactly as currently implemented,
      including the same output filenames used downstream.
  (2) Add a "conventional" TWFE benchmark (non-NYT) run, with minimal clutter:
        - model1a_lp_dynamic_betas_all_twfe.tsv
        - model1a_lp_window_betas_all_twfe.tsv
        - model1a_lp_pretrend_tests_all_twfe.tsv
  (3) Add a static conventional DiD coefficient (treated×post, TWFE) as an
      extra window-row inside the window TSVs:
        window = "did_post", a = 1, b = max_post_supported.

Design (NYT):
  - Mark treatment rows exactly as in the NYT windows.
  - Mark control rows exactly as in the NYT windows.
  - event_time = month_index - event_month_index for treated rows; event_time = -1 for all controls.
  - Regress ln(LP) on event_time dummies + unit FE + month FE (+ PortTr, + shocks).

Design (TWFE benchmark):
  - Uses broader windows (no NYT truncation of controls after later reforms).
  - Same regression specifications and output schema.
  - Outputs are written in separate pooled TSVs with suffix "_twfe".

Inference:
  - Main SEs are cluster-robust at the series level.
  - Pre-trend F-tests are computed using an OLS-style covariance for stability.

Outputs (NYT) written to: Thesis/Design/Output (new)/Model_1A
  - model1a_lp_dynamic_betas_all.tsv
  - model1a_lp_window_betas_all.tsv
  - model1a_lp_pretrend_tests_all.tsv
  - plus per-spec files (baseline/porttr/tr_shocks) as before.

Outputs (TWFE benchmark) written to the same folder (minimal extra clutter):
  - model1a_lp_dynamic_betas_all_twfe.tsv
  - model1a_lp_window_betas_all_twfe.tsv
  - model1a_lp_pretrend_tests_all_twfe.tsv
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ----------------------------------------------------------------------
# 0. Paths
# ----------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
THESIS_ROOT = THIS_FILE.parents[2]  # .../Thesis
LP_PANEL_PATH = THESIS_ROOT / "Data" / "LP" / "LP_Panel_monthly.tsv"

OUTPUT_DIR = THESIS_ROOT / "Design" / "Output (new)" / "Model_1A"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# 1. Helper data structures
# ----------------------------------------------------------------------

@dataclass
class Window:
    label: str              # e.g. "Haifa port", "Haifa-Bayport"
    start: Tuple[int, int]  # (year, month) inclusive
    end:   Tuple[int, int]  # (year, month) inclusive


@dataclass
class Spec:
    reform: str
    target: str
    event_year: int
    event_month: int
    treat_windows: List[Window]
    control_windows: List[Window]


@dataclass
class SpecWithFE:
    spec: Spec
    spec_name: str
    include_port_trends: bool = False
    include_shocks: bool = False


# ----------------------------------------------------------------------
# 2. Mapping from labels to LP_Panel filters
# ----------------------------------------------------------------------

LABEL_FILTERS: Dict[str, Dict[str, str]] = {
    # Port-level monthly series
    "Haifa port":  {"level": "port",    "port": "Haifa"},
    "Ashdod port": {"level": "port",    "port": "Ashdod"},

    # Terminal-level series (monthly-expanded from quarterly LP)
    "Haifa-Bayport": {"level": "terminal", "port": "Haifa",  "terminal": "Haifa-Bayport"},
    "Haifa-Legacy":  {"level": "terminal", "port": "Haifa",  "terminal": "Haifa-Legacy"},
    "Ashdod-HCT":    {"level": "terminal", "port": "Ashdod", "terminal": "Ashdod-HCT"},
    "Ashdod-Legacy": {"level": "terminal", "port": "Ashdod", "terminal": "Ashdod-Legacy"},
}


# ----------------------------------------------------------------------
# 3A. NYT specs (current "fixed" design: Haifa reform clocks only)
# ----------------------------------------------------------------------

NYT_SPECS: List[Spec] = [
    # Haifa competition entry (Bayport opens 09-2021)
    # Truncate at 10-2022 so Ashdod controls are excluded after Ashdod entry (11-2022).
    Spec(
        reform="haifa_comp",
        target="Haifa-Bayport terminal",
        event_year=2021,
        event_month=9,
        treat_windows=[
            Window("Haifa port",    (2019, 9), (2021, 8)),
            Window("Haifa-Bayport", (2021, 9), (2022, 10)),
        ],
        control_windows=[
            Window("Ashdod port",   (2019, 9), (2021, 7)),
            Window("Ashdod-Legacy", (2021, 8), (2022, 10)),
            Window("Ashdod-HCT",    (2021, 8), (2022, 10)),
        ],
    ),
    Spec(
        reform="haifa_comp",
        target="Haifa-Legacy terminal",
        event_year=2021,
        event_month=9,
        treat_windows=[
            Window("Haifa port",   (2019, 9), (2021, 8)),
            Window("Haifa-Legacy", (2021, 9), (2022, 10)),
        ],
        control_windows=[
            Window("Ashdod port",   (2019, 9), (2021, 7)),
            Window("Ashdod-Legacy", (2021, 8), (2022, 10)),
            Window("Ashdod-HCT",    (2021, 8), (2022, 10)),
        ],
    ),

    # Haifa privatization (sale 01-2023), sample exactly 01-2022..09-2023
    Spec(
        reform="haifa_priv",
        target="Haifa-Legacy terminal",
        event_year=2023,
        event_month=1,
        treat_windows=[
            Window("Haifa-Legacy", (2022, 1), (2023, 9)),
        ],
        control_windows=[
            Window("Haifa-Bayport", (2022, 1), (2023, 9)),
            Window("Ashdod-Legacy", (2022, 1), (2023, 9)),
            Window("Ashdod-HCT",    (2022, 1), (2023, 9)),
        ],
    ),
]


# ----------------------------------------------------------------------
# 3B. TWFE benchmark specs (conventional; no NYT truncation)
# Minimal set: same three regressions, but extend the post as far as the data allow.
# We use (2099,12) as an "end-of-sample" sentinel and clamp to last available month.
# ----------------------------------------------------------------------

TWFE_SPECS: List[Spec] = [
    # Haifa competition: no truncation at 10-2022; keep Ashdod controls beyond 11-2022.
    Spec(
        reform="haifa_comp",
        target="Haifa-Bayport terminal",
        event_year=2021,
        event_month=9,
        treat_windows=[
            Window("Haifa port",    (2019, 9), (2021, 8)),
            Window("Haifa-Bayport", (2021, 9), (2099, 12)),
        ],
        control_windows=[
            Window("Ashdod port",   (2019, 9), (2021, 7)),
            Window("Ashdod-Legacy", (2021, 8), (2099, 12)),
            Window("Ashdod-HCT",    (2021, 8), (2099, 12)),
        ],
    ),
    Spec(
        reform="haifa_comp",
        target="Haifa-Legacy terminal",
        event_year=2021,
        event_month=9,
        treat_windows=[
            Window("Haifa port",   (2019, 9), (2021, 8)),
            Window("Haifa-Legacy", (2021, 9), (2099, 12)),
        ],
        control_windows=[
            Window("Ashdod port",   (2019, 9), (2021, 7)),
            Window("Ashdod-Legacy", (2021, 8), (2099, 12)),
            Window("Ashdod-HCT",    (2021, 8), (2099, 12)),
        ],
    ),

    # Haifa privatization: extend beyond 09-2023 as far as data allow (terminals only).
    Spec(
        reform="haifa_priv",
        target="Haifa-Legacy terminal",
        event_year=2023,
        event_month=1,
        treat_windows=[
            Window("Haifa-Legacy", (2022, 1), (2099, 12)),
        ],
        control_windows=[
            Window("Haifa-Bayport", (2022, 1), (2099, 12)),
            Window("Ashdod-Legacy", (2022, 1), (2099, 12)),
            Window("Ashdod-HCT",    (2022, 1), (2099, 12)),
        ],
    ),
]


# ----------------------------------------------------------------------
# 4. Optional shock controls
# ----------------------------------------------------------------------

EXPLICIT_SHOCK_COLS: List[str] = ["covid_shock", "war_shock"]


def get_shock_control_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in EXPLICIT_SHOCK_COLS if c in df.columns]


# ----------------------------------------------------------------------
# 5. Load panel and helpers
# ----------------------------------------------------------------------

def load_lp_panel(path: Path) -> pd.DataFrame:
    print(f"Reading monthly LP panel from: {path}")
    df = pd.read_csv(path, sep="\t")
    print(f"Loaded {len(df)} rows from LP_Panel_monthly.tsv.")

    required_cols = {"year", "month", "month_index", "LP",
                     "series_id", "level", "freq", "port", "terminal"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"LP_Panel_monthly.tsv is missing required columns: {missing}")

    for col in ["year", "month", "month_index"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if (df["LP"] <= 0).any():
        raise ValueError("LP contains non-positive values; cannot take logs.")
    df["log_LP"] = np.log(df["LP"])

    # Simple shock controls
    covid_mask = df["year"].between(2020, 2021, inclusive="both").fillna(False)
    df["covid_shock"] = covid_mask.astype(int)

    war_mask = ((df["year"] > 2023) | ((df["year"] == 2023) & (df["month"] >= 10))).fillna(False)
    df["war_shock"] = war_mask.astype(int)

    # FE and clustering IDs
    df["unit_id"] = df["series_id"]
    df["time_id"] = df["month_index"]
    df["cluster_id"] = df["series_id"]

    return df


def build_year_month_to_index(df: pd.DataFrame) -> Dict[Tuple[int, int], int]:
    mapping: Dict[Tuple[int, int], int] = {}
    grouped = df.groupby(["year", "month"])["month_index"].unique()
    for (y, m), idxs in grouped.items():
        if len(idxs) != 1:
            raise ValueError(f"(year={y}, month={m}) has multiple month_index values: {idxs}")
        mapping[(y, m)] = int(idxs[0])
    print(f"Built (year, month) -> month_index mapping for {len(mapping)} months.")
    return mapping


def _ym_serial(ym: Tuple[int, int]) -> int:
    y, m = ym
    return int(y) * 12 + int(m)


def year_month_to_index_strict(mapping: Dict[Tuple[int, int], int], ym: Tuple[int, int]) -> int:
    if ym not in mapping:
        raise KeyError(f"(year={ym[0]}, month={ym[1]}) not found in mapping.")
    return mapping[ym]


def year_month_to_index_clamped(mapping: Dict[Tuple[int, int], int], ym: Tuple[int, int]) -> int:
    """If ym isn't present, clamp to the latest available month <= ym."""
    if ym in mapping:
        return mapping[ym]

    target = _ym_serial(ym)
    keys = list(mapping.keys())
    serials = np.array([_ym_serial(k) for k in keys], dtype=int)

    ok = serials <= target
    if not ok.any():
        raise KeyError(f"No available month <= (year={ym[0]}, month={ym[1]}) in mapping.")
    best_key = keys[int(serials[ok].argmax())]
    return mapping[best_key]


def get_series_id_for_label(df: pd.DataFrame, label: str) -> str:
    if label not in LABEL_FILTERS:
        raise KeyError(f"Label '{label}' not found in LABEL_FILTERS.")

    conds = LABEL_FILTERS[label]
    mask = pd.Series(True, index=df.index)
    for col, val in conds.items():
        mask &= (df[col] == val)

    sids = df.loc[mask, "series_id"].unique()
    if len(sids) == 0:
        unique = df[["series_id", "level", "freq", "port", "terminal"]].drop_duplicates()
        raise ValueError(
            f"No series_id found for label '{label}' with filters {conds}.\n"
            f"Some unique series in LP_Panel:\n{unique.head(20)}"
        )
    if len(sids) > 1:
        raise ValueError(
            f"Label '{label}' matched multiple series_ids: {sids}. Refine LABEL_FILTERS."
        )
    return str(sids[0])


# ----------------------------------------------------------------------
# 6. Build estimation sample for one Spec
# ----------------------------------------------------------------------

def build_es_sample(
    df: pd.DataFrame,
    spec: Spec,
    ym_to_idx: Dict[Tuple[int, int], int],
    clamp_windows: bool,
) -> pd.DataFrame:
    """
    Build the event-study sample for one Spec.

    If clamp_windows=True, any window end date beyond the data range is clamped
    to the last available month <= requested (used for TWFE sentinel windows).
    """
    df_es = df.copy()
    df_es["in_sample"] = False
    df_es["treated"] = False

    ym_to_index = year_month_to_index_clamped if clamp_windows else year_month_to_index_strict

    # 1) Mark treatment windows
    for w in spec.treat_windows:
        sid = get_series_id_for_label(df_es, w.label)
        start_idx = ym_to_index(ym_to_idx, w.start)
        end_idx = ym_to_index(ym_to_idx, w.end)

        mask = (df_es["series_id"] == sid) & df_es["month_index"].between(start_idx, end_idx)
        df_es.loc[mask, "in_sample"] = True
        df_es.loc[mask, "treated"] = True

    # 2) Mark control windows
    for w in spec.control_windows:
        sid = get_series_id_for_label(df_es, w.label)
        start_idx = ym_to_index(ym_to_idx, w.start)
        end_idx = ym_to_index(ym_to_idx, w.end)

        mask = (df_es["series_id"] == sid) & df_es["month_index"].between(start_idx, end_idx)
        df_es.loc[mask, "in_sample"] = True

    df_es = df_es[df_es["in_sample"]].copy()

    # 3) event_time
    event_index = ym_to_index(ym_to_idx, (spec.event_year, spec.event_month))

    df_es["event_time"] = -1
    treated_mask = df_es["treated"]
    df_es.loc[treated_mask, "event_time"] = df_es.loc[treated_mask, "month_index"] - event_index

    # 4) infer port from series_id prefix
    df_es["port"] = df_es["series_id"].str.extract(r"^(Haifa|Ashdod)", expand=False)
    if df_es["port"].isna().any():
        bad = df_es.loc[df_es["port"].isna(), "series_id"].unique()
        raise ValueError(f"Could not infer port from series_id for: {bad}")

    # 5) trend
    df_es["time_index"] = df_es["time_id"]
    df_es["port_trend"] = 0.0
    for port_name in df_es["port"].unique():
        mask_port = df_es["port"] == port_name
        slope_sign = 1.0 if port_name == "Haifa" else -1.0
        df_es.loc[mask_port, "port_trend"] = df_es.loc[mask_port, "time_index"] * slope_sign

    # Convenience: applied notation label
    df_es["j"] = df_es["event_time"]

    return df_es


# ----------------------------------------------------------------------
# 7. Regression + coefficient extraction
# ----------------------------------------------------------------------

def _parse_event_time_from_param_name(name: str) -> Optional[int]:
    prefix = "C(event_time, Treatment(reference=-1))[T."
    if not name.startswith(prefix):
        return None
    m_str = name[len(prefix):].rstrip("]")
    try:
        return int(m_str)
    except ValueError:
        return None


def run_event_study(df_es: pd.DataFrame, spec_with_fe: SpecWithFE, shock_cols: List[str]):
    df_es = df_es.copy()

    # Coerce Int64 -> int64 for patsy
    for col in ["event_time", "time_id"]:
        if pd.api.types.is_integer_dtype(df_es[col].dtype):
            df_es[col] = df_es[col].astype("int64")

    formula = "log_LP ~ C(event_time, Treatment(reference=-1)) + C(unit_id) + C(time_id)"
    if spec_with_fe.include_port_trends:
        formula += " + port_trend"

    used_shocks: List[str] = []
    if spec_with_fe.include_shocks and shock_cols:
        used_shocks = [c for c in shock_cols if c in df_es.columns]
        if used_shocks:
            formula += " + " + " + ".join(used_shocks)

    # N(j) counts
    n_by_event_time = df_es.groupby("event_time")["unit_id"].size().to_dict()

    model = smf.ols(formula=formula, data=df_es)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df_es["cluster_id"]})

    return result, n_by_event_time


def run_static_did(df_es: pd.DataFrame, spec_with_fe: SpecWithFE, shock_cols: List[str]):
    """
    Conventional DiD: TWFE regression with a single treated×post indicator.
    We define post as j>=1 to align with your "post" windows in the tables.
    """
    df_did = df_es.copy()

    df_did["post_ge_1"] = (df_did["event_time"] >= 1).astype(int)
    df_did["treated_int"] = df_did["treated"].astype(int)
    df_did["treated_post"] = df_did["treated_int"] * df_did["post_ge_1"]

    # Coerce Int64 -> int64 for patsy categories
    for col in ["time_id"]:
        if pd.api.types.is_integer_dtype(df_did[col].dtype):
            df_did[col] = df_did[col].astype("int64")

    formula = "log_LP ~ treated_post + C(unit_id) + C(time_id)"
    if spec_with_fe.include_port_trends:
        formula += " + port_trend"

    used_shocks: List[str] = []
    if spec_with_fe.include_shocks and shock_cols:
        used_shocks = [c for c in shock_cols if c in df_did.columns]
        if used_shocks:
            formula += " + " + " + ".join(used_shocks)

    model = smf.ols(formula=formula, data=df_did)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df_did["cluster_id"]})

    beta = float(res.params.get("treated_post", np.nan))
    se = float(res.bse.get("treated_post", np.nan))
    pval = float(res.pvalues.get("treated_post", np.nan))

    return beta, se, pval, float(res.rsquared), int(res.nobs)


def extract_dynamic_betas(result, spec_with_fe: SpecWithFE, n_by_event_time: Dict[int, int]) -> pd.DataFrame:
    spec = spec_with_fe.spec
    rows = []

    params = result.params
    bse = result.bse
    pvals = result.pvalues

    for name, beta in params.items():
        j = _parse_event_time_from_param_name(name)
        if j is None:
            continue

        se = float(bse.get(name, np.nan))
        t_stat = float(beta / se) if (np.isfinite(se) and se != 0) else np.nan
        pval = float(pvals.get(name, np.nan))

        rows.append(
            {
                "reform": spec.reform,
                "target": spec.target,
                "spec_name": spec_with_fe.spec_name,
                "event_time": int(j),
                "j": int(j),
                "beta": float(beta),
                "se": se,
                "t": t_stat,
                "pvalue": pval,
                "n_event_obs": float(n_by_event_time.get(int(j), np.nan)),
                "n_obs": int(result.nobs),
                "r2": float(result.rsquared),
            }
        )

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 8. Window averages from event-study betas
# ----------------------------------------------------------------------

PRETREND_MIN = -12
PRETREND_MAX = -2

WINDOWS: Dict[str, Tuple[int, int]] = {
    "avg_pre":   (PRETREND_MIN, PRETREND_MAX),
    "post_1yr":  (1, 12),
    "post_2yrs": (1, 24),
    "full_post": (1, 999),  # interpreted as [1, max observed post j]
}

PRETREND_BINS: List[Tuple[int, int]] = [
    (PRETREND_MIN, -7),
    (-6, PRETREND_MAX),
]


def compute_window_averages(result, spec_with_fe: SpecWithFE) -> pd.DataFrame:
    spec = spec_with_fe.spec
    params = result.params
    cov = result.cov_params()

    j_to_name: Dict[int, str] = {}
    for name in params.index:
        j = _parse_event_time_from_param_name(name)
        if j is None:
            continue
        j_to_name[int(j)] = name

    if not j_to_name:
        return pd.DataFrame([])

    available_js = sorted(j_to_name.keys())
    max_post_j = max((j for j in available_js if j > 0), default=None)

    rows = []
    for wname, (a, b) in WINDOWS.items():
        b_eff = b
        if max_post_j is not None:
            if b == 999:
                b_eff = int(max_post_j)
            elif a >= 1:
                b_eff = int(min(b, max_post_j))

        js = [j for j in available_js if (j >= a and j <= b_eff)]
        if not js:
            continue

        w = pd.Series(0.0, index=params.index)
        weight = 1.0 / len(js)
        for j in js:
            w[j_to_name[j]] = weight

        beta_w = float(np.dot(w.values, params.values))
        var_w = float(np.dot(w.values, np.dot(cov.values, w.values)))
        se_w = float(np.sqrt(var_w)) if var_w >= 0 else np.nan

        rows.append(
            {
                "reform": spec.reform,
                "target": spec.target,
                "spec_name": spec_with_fe.spec_name,
                "window": wname,
                "a": int(a),
                "b": int(b_eff),
                "beta": beta_w,
                "se": se_w,
                "n_obs": int(result.nobs),
                "r2": float(result.rsquared),
            }
        )

    return pd.DataFrame(rows)


def compute_did_window_row(
    df_es: pd.DataFrame,
    spec_with_fe: SpecWithFE,
    beta: float,
    se: float,
    r2: float,
    nobs: int,
) -> pd.DataFrame:
    """Store the static DiD estimate as an extra window row (window='did_post')."""
    spec = spec_with_fe.spec
    treated_post = df_es.loc[df_es["treated"] & (df_es["event_time"] > 0), "event_time"]
    max_post = int(treated_post.max()) if len(treated_post) else 999

    return pd.DataFrame(
        [
            {
                "reform": spec.reform,
                "target": spec.target,
                "spec_name": spec_with_fe.spec_name,
                "window": "did_post",
                "a": 1,
                "b": max_post,
                "beta": float(beta),
                "se": float(se),
                "n_obs": int(nobs),
                "r2": float(r2),
            }
        ]
    )


# ----------------------------------------------------------------------
# 9. Pre-trend F-test (coarse bins, OLS-style covariance)
# ----------------------------------------------------------------------

def compute_pretrend_f_test(result, spec_with_fe: SpecWithFE) -> pd.DataFrame:
    params = result.params
    names = list(params.index)
    k = len(names)

    event_param_info: List[Tuple[int, int]] = []
    for idx, name in enumerate(names):
        j = _parse_event_time_from_param_name(name)
        if j is None:
            continue
        if PRETREND_MIN <= j <= PRETREND_MAX:
            event_param_info.append((int(j), idx))

    if not event_param_info:
        return pd.DataFrame([])

    R_rows: List[np.ndarray] = []
    bins_used: List[Tuple[int, int]] = []
    for (a, b) in PRETREND_BINS:
        idxs = [idx for (j, idx) in event_param_info if (j >= a and j <= b)]
        if not idxs:
            continue
        r = np.zeros(k)
        w = 1.0 / len(idxs)
        for j_idx in idxs:
            r[j_idx] = w
        R_rows.append(r)
        bins_used.append((a, b))

    if not R_rows:
        return pd.DataFrame([])

    R = np.vstack(R_rows)
    n_restr = int(R.shape[0])

    try:
        cov_unclustered = np.asarray(result.normalized_cov_params) * float(result.mse_resid)
        wtest = result.wald_test(R, cov_p=cov_unclustered, use_f=True, scalar=False)

        f_raw = getattr(wtest, "fvalue", getattr(wtest, "statistic", np.nan))
        f_val = float(np.asarray(f_raw).ravel()[0])

        p_raw = getattr(wtest, "pvalue", np.nan)
        p_val = float(np.asarray(p_raw).ravel()[0])

        df_num_attr = getattr(wtest, "df_num", None)
        df_denom_attr = getattr(wtest, "df_denom", None)
        df_num = float(df_num_attr) if df_num_attr is not None else float(n_restr)
        df_denom = float(df_denom_attr) if df_denom_attr is not None else float(result.df_resid)
    except Exception:
        f_val = np.nan
        p_val = np.nan
        df_num = float(n_restr)
        df_denom = float(result.df_resid)

    return pd.DataFrame(
        [
            {
                "reform": spec_with_fe.spec.reform,
                "target": spec_with_fe.spec.target,
                "spec": spec_with_fe.spec_name,
                "pre_min": float(PRETREND_MIN),
                "pre_max": float(PRETREND_MAX),
                "n_leads_total": float(len(event_param_info)),
                "n_bins_defined": float(len(PRETREND_BINS)),
                "n_bins_used": float(len(bins_used)),
                "f_stat": f_val,
                "pvalue": p_val,
                "df_num": df_num,
                "df_denom": df_denom,
                "n_obs": int(result.nobs),
                "r2": float(result.rsquared),
            }
        ]
    )


# ----------------------------------------------------------------------
# 10. Driver helpers
# ----------------------------------------------------------------------

def build_specs_with_fe(base_specs: List[Spec], shock_cols_all: List[str]) -> List[SpecWithFE]:
    out: List[SpecWithFE] = []
    for s in base_specs:
        out.append(SpecWithFE(spec=s, spec_name="baseline", include_port_trends=False, include_shocks=False))
        out.append(SpecWithFE(spec=s, spec_name="porttr", include_port_trends=True, include_shocks=False))
        if shock_cols_all:
            out.append(SpecWithFE(spec=s, spec_name="tr_shocks", include_port_trends=True, include_shocks=True))
    return out


def run_design(
    df: pd.DataFrame,
    ym_to_idx: Dict[Tuple[int, int], int],
    shock_cols_all: List[str],
    base_specs: List[Spec],
    design_name: str,
    clamp_windows: bool,
    write_per_spec: bool,
    suffix: str,
) -> None:
    """
    Run a full set of specs and write outputs.

    For NYT: suffix="" and write_per_spec=True (preserves current behavior).
    For TWFE: suffix="_twfe" and write_per_spec=False (minimal clutter).
    """
    print(f"\n==================== {design_name} run ====================")
    specs_with_fe = build_specs_with_fe(base_specs, shock_cols_all)

    dynamic_by_spec: Dict[str, List[pd.DataFrame]] = {}
    windows_by_spec: Dict[str, List[pd.DataFrame]] = {}
    pretrend_by_spec: Dict[str, List[pd.DataFrame]] = {}

    for spec_with_fe in specs_with_fe:
        spec = spec_with_fe.spec
        print(f"\n=== [{design_name}] reform={spec.reform}, target={spec.target}, spec={spec_with_fe.spec_name} ===")

        df_es = build_es_sample(df, spec, ym_to_idx, clamp_windows=clamp_windows)

        if df_es["treated"].sum() == 0:
            print(f"[WARN] No treated observations for reform={spec.reform}, target={spec.target}. Skipping.")
            continue

        n_treat = int(df_es["treated"].sum())
        n_ctrl = int(len(df_es) - n_treat)
        print(f"Sample size: {len(df_es)} rows ({n_treat} treated, {n_ctrl} controls).")

        # Event-study regression
        es_res, n_by_j = run_event_study(df_es, spec_with_fe, shock_cols_all)

        dyn = extract_dynamic_betas(es_res, spec_with_fe, n_by_j)
        win = compute_window_averages(es_res, spec_with_fe)
        pre = compute_pretrend_f_test(es_res, spec_with_fe)

        # Static DiD regression (treated×post), stored as window='did_post'
        did_beta, did_se, did_p, did_r2, did_nobs = run_static_did(df_es, spec_with_fe, shock_cols_all)
        did_row = compute_did_window_row(df_es, spec_with_fe, did_beta, did_se, did_r2, did_nobs)
        if not win.empty:
            win = pd.concat([win, did_row], ignore_index=True)
        else:
            win = did_row

        if not dyn.empty:
            dynamic_by_spec.setdefault(spec_with_fe.spec_name, []).append(dyn)
        if not win.empty:
            windows_by_spec.setdefault(spec_with_fe.spec_name, []).append(win)
        if not pre.empty:
            pretrend_by_spec.setdefault(spec_with_fe.spec_name, []).append(pre)

    base_name = "model1a_lp"

    # Write per-spec (NYT only)
    all_dyn: List[pd.DataFrame] = []
    all_win: List[pd.DataFrame] = []
    all_pre: List[pd.DataFrame] = []

    if write_per_spec:
        for spec_name, frames in dynamic_by_spec.items():
            ddf = pd.concat(frames, ignore_index=True)
            all_dyn.append(ddf)
            path = OUTPUT_DIR / f"{base_name}_dynamic_betas_{spec_name}{suffix}.tsv"
            ddf.to_csv(path, sep="\t", index=False)
            print(f"Saved dynamic betas ({design_name}, spec={spec_name}) to: {path}")

        for spec_name, frames in windows_by_spec.items():
            wdf = pd.concat(frames, ignore_index=True)
            all_win.append(wdf)
            path = OUTPUT_DIR / f"{base_name}_window_betas_{spec_name}{suffix}.tsv"
            wdf.to_csv(path, sep="\t", index=False)
            print(f"Saved window betas ({design_name}, spec={spec_name}) to: {path}")

        for spec_name, frames in pretrend_by_spec.items():
            pdf = pd.concat(frames, ignore_index=True)
            all_pre.append(pdf)
            path = OUTPUT_DIR / f"{base_name}_pretrend_tests_{spec_name}{suffix}.tsv"
            pdf.to_csv(path, sep="\t", index=False)
            print(f"Saved pretrend tests ({design_name}, spec={spec_name}) to: {path}")

    # Always write pooled "all specs" files
    if dynamic_by_spec:
        if not all_dyn:
            all_dyn = [pd.concat(frames, ignore_index=True) for frames in dynamic_by_spec.values()]
        dyn_all = pd.concat(all_dyn, ignore_index=True)
        path = OUTPUT_DIR / f"{base_name}_dynamic_betas_all{suffix}.tsv"
        dyn_all.to_csv(path, sep="\t", index=False)
        print(f"Saved pooled dynamic betas ({design_name}) to: {path}")

    if windows_by_spec:
        if not all_win:
            all_win = [pd.concat(frames, ignore_index=True) for frames in windows_by_spec.values()]
        win_all = pd.concat(all_win, ignore_index=True)
        path = OUTPUT_DIR / f"{base_name}_window_betas_all{suffix}.tsv"
        win_all.to_csv(path, sep="\t", index=False)
        print(f"Saved pooled window betas ({design_name}) to: {path}")

    if pretrend_by_spec:
        if not all_pre:
            all_pre = [pd.concat(frames, ignore_index=True) for frames in pretrend_by_spec.values()]
        pre_all = pd.concat(all_pre, ignore_index=True)
        path = OUTPUT_DIR / f"{base_name}_pretrend_tests_all{suffix}.tsv"
        pre_all.to_csv(path, sep="\t", index=False)
        print(f"Saved pooled pretrend tests ({design_name}) to: {path}")


def main() -> None:
    df = load_lp_panel(LP_PANEL_PATH)
    ym_to_idx = build_year_month_to_index(df)
    shock_cols_all = get_shock_control_cols(df)

    if shock_cols_all:
        print(f"Detected shock-control columns: {shock_cols_all}")
    else:
        print("No shock-control columns detected; '+Tr&Shocks' specs will be skipped.")

    # 1) NYT run (exactly preserves your current output filenames)
    run_design(
        df=df,
        ym_to_idx=ym_to_idx,
        shock_cols_all=shock_cols_all,
        base_specs=NYT_SPECS,
        design_name="NYT",
        clamp_windows=False,
        write_per_spec=True,
        suffix="",
    )

    # 2) TWFE benchmark run (minimal extra clutter: pooled outputs only)
    run_design(
        df=df,
        ym_to_idx=ym_to_idx,
        shock_cols_all=shock_cols_all,
        base_specs=TWFE_SPECS,
        design_name="TWFE",
        clamp_windows=True,          # to handle the (2099,12) sentinel
        write_per_spec=False,        # reduce clutter
        suffix="_twfe",
    )


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# DIAGNOSTIC NOTE (kept from v2; revisit once true monthly hours arrive)
#
# With small treated samples + heavy FE saturation, cluster-robust SEs can be unstable.
# Treat inference diagnostics cautiously until true labor-hours data is incorporated.
# ----------------------------------------------------------------------


# =============================================================================
# MODEL_1A(v3) RUN EVALUATION (2026-03-xx)
#
# Summary:
# - v3 ran successfully and produced the intended NYT + TWFE outputs with minimal
#   additional clutter. NYT outputs preserved legacy filenames; TWFE wrote only
#   pooled *_all_twfe.tsv files (dynamic/window/pretrend).
#
# Console-validated sample sizes:
# - NYT:
#   * haifa_comp (Bayport target): 91 rows (38 treated, 53 controls)
#   * haifa_comp (Legacy target):  91 rows (38 treated, 53 controls)
#   * haifa_priv (Legacy target):  84 rows (21 treated, 63 controls)
# - TWFE (conventional benchmark, extended windows; no NYT truncation):
#   * haifa_comp targets:          169 rows (64 treated, 105 controls)
#   * haifa_priv (Legacy target):  144 rows (36 treated, 108 controls)
#
# Output integrity checks (from exported TSVs):
# - Window betas files (NYT and TWFE) contain 45 rows each (= 9 spec-combos x 5 windows),
#   with windows: avg_pre, post_1yr, post_2yrs, full_post, did_post.
# - Event-time coefficients are contiguous (no missing j's) and window bounds (b) match
#   max available post-j for each (reform,target,spec_name).
# - Pretrend test TSVs contain 9 rows each (3 targets x 3 specs) with non-missing F-stats/p-values.
#
# Warnings / expected numerical issues:
# - statsmodels "invalid value encountered in sqrt" appeared in some clustered runs.
#   This is consistent with small-cluster / high-FE designs and does not imply a
#   logic error in sample construction.
# - One isolated downstream effect: the static DiD ("did_post") SE is NaN for
#   NYT / haifa_comp / Haifa-Bayport / baseline (only). Dynamic ES betas still have
#   finite SEs. If needed, implement a fallback for the *static DiD only*:
#     - if se is NaN under clustered SE, re-fit the DiD regression with cov_type="HC1".
#
# Net:
# - No code-logic bugs detected; outputs are structurally correct and ready for the
#   next step (tables/plots and Model_1B alignment).
# =============================================================================