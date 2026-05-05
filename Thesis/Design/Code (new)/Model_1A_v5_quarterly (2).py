#!/usr/bin/env python3
"""
Model_1A_v5_quarterly.py

Quarterly rewrite of Model 1A for the Israel ports thesis.

Why this file exists
--------------------
The previous Model 1A(v4) operated on LP_Panel_monthly.tsv, which is convenient for a
monthly event-study shell but can be misleading because part of the terminal-level LP signal
is only genuinely observed quarterly. In the previous pipeline, quarterly terminal LP was
expanded into within-quarter monthly step functions purely for alignment convenience. That
monthly expansion made it easy for the regressions to *look* data-rich while still carrying
quarterly information content.

The main empirical symptoms were:
  * saturated or near-saturated event-study regressions,
  * within-R^2 values at or near 1,
  * undefined or fragile pretrend tests,
  * effectively zero standard errors in some competition specifications,
  * dynamic paths that reflected pseudo-monthly duplication rather than real monthly movement.

This new file implements the changes we discussed after reviewing the v4 outputs:

1. It reads the mixed-frequency LP panel and constructs a QUARTERLY analysis panel.
   - Pre-reform monthly PORT LP is honestly collapsed to quarters.
   - Post-reform TERMINAL LP stays at its native quarterly frequency.
   - This is the single biggest design change. The goal is not to make the model "more relaxed"
     in the abstract. The goal is to match econometric frequency to information frequency.

2. It keeps the same broad thesis architecture:
   - NYT-style local event-study objects where the comparison path is truly not-yet-treated
     in the relevant window when that is feasible.
   - Conventional DiD / TWFE-style benchmark panels retained for comparison because the thesis
     still wants those columns and the advisor explicitly asked that they remain.

3. It reduces dynamic dimensionality.
   - Instead of a dense monthly lead/lag path, it estimates a small number of coarse QUARTERLY
     event-time bins. This is meant to reduce saturation and to make pretrend testing defined
     more often.

4. It makes the trend-based specification explicit.
   - The old code had a "porttr" specification. In this rewrite, that idea is retained but made
     transparent: the baseline specification uses quarter fixed effects; the trend specification
     replaces the full quarter-FE structure with a linear time trend (and port-specific trends
     when multiple ports are in the sample).
   - This is retained as a benchmark / robustness comparison, NOT as the preferred causal object.

5. It uses heteroskedasticity-robust HC1 standard errors by default.
   - With tiny comparison sets (often only two paths), cluster-robust inference at the unit level
     is not reliable. The point here is to get technically informative regressions first. If the
     results later justify a different inference layer, that can be changed separately.

Important scope notes
---------------------
This file is still written around the thesis's established Model 1A logic:
  * competition focuses on incumbent paths,
  * Haifa competition has an NYT object,
  * Ashdod competition is benchmark/TWFE only,
  * Haifa privatization is legacy-centered with Bayport retained as a placebo/comparison path.

Outputs
-------
The file writes quarterly analogues of the familiar pooled result files:
  * model1a_q_lp_dynamic_betas_all.tsv
  * model1a_q_lp_dynamic_betas_all_twfe.tsv
  * model1a_q_lp_window_betas_all.tsv
  * model1a_q_lp_window_betas_all_twfe.tsv
  * model1a_q_lp_pretrend_tests_all.tsv
  * model1a_q_lp_pretrend_tests_all_twfe.tsv
  * model1a_q_lp_static_betas_all_twfe.tsv
  * model1a_q_analysis_panel.tsv
  * model1a_q_manifest.json

This script is intentionally verbose and heavily commented because it is meant to be a
transparent design pivot away from the v4 monthly shell rather than a quiet patch.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy


# ============================================================================
# Quarter utilities
# ============================================================================

def quarter_from_month(month: int) -> int:
    return ((int(month) - 1) // 3) + 1


def parse_quarter_value(x) -> int:
    """
    Accept quarter values stored as 1/2/3/4, '1'/'2'/..., or 'Q1'/'Q2'/... .
    The active LP panels sometimes store quarter labels in the human-readable Q# form.
    """
    if pd.isna(x):
        raise ValueError("Quarter value is missing")
    s = str(x).strip()
    if s.upper().startswith("Q"):
        s = s[1:]
    return int(float(s))


def quarter_id(year: int, quarter: int) -> int:
    """Monotone integer quarter index for sorting and event-time arithmetic."""
    return int(year) * 4 + int(quarter) - 1


def quarter_label(year: int, quarter: int) -> str:
    return f"{int(year)}Q{int(quarter)}"


# ============================================================================
# Canonical naming helpers
# ============================================================================

def norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = s.replace("—", "-")
    s = s.replace("–", "-")
    s = s.replace("_", "-")
    s = s.replace("  ", " ")
    return s


def canonical_terminal(port: str, terminal: str, series_id: str) -> str:
    p = norm_text(port)
    t = norm_text(terminal)
    sid = norm_text(series_id)

    if "haifa" in sid and "bay" in sid:
        return "Haifa-Bayport"
    if "haifa" in sid and "legacy" in sid:
        return "Haifa-Legacy"
    if "ashdod" in sid and "hct" in sid:
        return "Ashdod-HCT"
    if "ashdod" in sid and "legacy" in sid:
        return "Ashdod-Legacy"

    if p == "haifa":
        if "sipg" in t or "bay" in t:
            return "Haifa-Bayport"
        if "legacy" in t:
            return "Haifa-Legacy"
    if p == "ashdod":
        if "hct" in t:
            return "Ashdod-HCT"
        if "legacy" in t:
            return "Ashdod-Legacy"

    return terminal if terminal not in (None, "", np.nan) else ""


def canonical_series_key(port: str, level: str, terminal: str, series_id: str) -> Optional[str]:
    p = norm_text(port)
    lev = norm_text(level)
    term = canonical_terminal(port, terminal, series_id)

    if lev == "port":
        if p == "haifa":
            return "haifa_port"
        if p == "ashdod":
            return "ashdod_port"
        return None

    # terminal-level mapping
    if term == "Haifa-Legacy":
        return "haifa_legacy"
    if term == "Haifa-Bayport":
        return "haifa_bayport"
    if term == "Ashdod-Legacy":
        return "ashdod_legacy"
    if term == "Ashdod-HCT":
        return "ashdod_hct"

    return None


# ============================================================================
# Input panel construction
# ============================================================================

def resolve_lp_path(user_path: Optional[str] = None) -> Path:
    """
    Resolve the LP panel path in a way that is explicit about the new pipeline.

    Priority order:
      1. A user-supplied path via --lp
      2. The version-specific common_rule_v5 LP panel
      3. The canonical active LP panel

    This matters because the whole thesis runner builds version-specific outputs first and then
    promotes them into canonical active locations. When running this file standalone, we prefer the
    version-specific v5 LP panel when it exists so there is no ambiguity about which labor-proxy build
    is being used downstream.
    """
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Requested LP panel does not exist: {p}")

    candidates = [
        Path('Data/LP/common_rule_v5/LP_Panel.tsv'),
        Path('Data/LP/common_rule_v5/LP_Panel_commonrule_v5.tsv'),
        Path('Data/LP/LP_Panel.tsv'),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        'Could not find an LP panel. Looked for common_rule_v5 and canonical LP_Panel.tsv locations.'
    )

def build_quarterly_lp_panel(lp_path: Path) -> pd.DataFrame:
    """
    Construct the quarterly analysis panel from LP_Panel.tsv.

    Key design change relative to Model_1A(v4):
    ------------------------------------------
    v4 loaded LP_Panel_monthly.tsv, which contains pseudo-monthly rows for quarterly terminal
    LP. This rewrite instead starts from LP_Panel.tsv and builds a quarterly panel honestly.

    Implementation choice:
      * for pre-reform monthly port LP, we average LOG(LP) within quarter;
      * for post-reform quarterly terminal LP, we keep the native quarterly observation.

    Why average log(LP) rather than LP and log afterward?
      Because the regression outcome is log(LP). Averaging log(LP) across the three monthly
      port observations within a quarter makes the quarterly outcome live on the same scale as
      the regression object while avoiding an unnecessary nonlinear transformation after
      aggregation.
    """
    df = pd.read_csv(lp_path, sep="\t")

    required = {"port", "level", "series_id", "LP", "freq", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"LP panel is missing required columns: {sorted(missing)}")

    if "terminal" not in df.columns:
        df["terminal"] = ""
    if "quarter" not in df.columns:
        df["quarter"] = np.nan
    if "month" not in df.columns:
        df["month"] = np.nan

    df = df.copy()
    df = df[df["LP"].notna()].copy()
    df = df[df["LP"] > 0].copy()

    # Canonical identifiers.
    df["terminal_canon"] = [
        canonical_terminal(p, t, s) for p, t, s in zip(df["port"], df["terminal"], df["series_id"])
    ]
    df["series_key"] = [
        canonical_series_key(p, l, t, s)
        for p, l, t, s in zip(df["port"], df["level"], df["terminal"], df["series_id"])
    ]
    df = df[df["series_key"].notna()].copy()

    df["log_LP"] = np.log(df["LP"].astype(float))

    out_rows: List[pd.DataFrame] = []

    port_keys = {"haifa_port", "ashdod_port"}
    term_keys = {"haifa_legacy", "haifa_bayport", "ashdod_legacy", "ashdod_hct"}

    # ---------------------------------------------------------------------
    # Monthly port LP -> quarterly port LP (average log(LP) within quarter).
    # ---------------------------------------------------------------------
    # Important robustness change:
    # We do NOT rely only on literal freq labels here because the active LP panel can come either
    # from the canonical promoted file or directly from the version-specific v5 folder, and small
    # schema-label differences are possible across builds. Instead, we use the economically meaningful
    # series keys plus the presence of a month field.
    port_month = df[df["series_key"].isin(port_keys) & df["month"].notna()].copy()
    if not port_month.empty:
        port_month["month"] = port_month["month"].astype(int)
        port_month["quarter"] = port_month["month"].apply(quarter_from_month)
        q = (
            port_month.groupby(["series_key", "port", "year", "quarter"], as_index=False)
            .agg(log_LP=("log_LP", "mean"), n_obs=("log_LP", "size"))
        )
        q["freq"] = "quarterly_from_monthly"
        q["level"] = "port"
        q["terminal"] = ""
        out_rows.append(q)
    else:
        # Fallback: if the port rows are already quarter-coded in the LP panel, keep them directly.
        port_q_direct = df[df["series_key"].isin(port_keys) & df["quarter"].notna()].copy()
        if not port_q_direct.empty:
            port_q_direct["quarter"] = port_q_direct["quarter"].apply(parse_quarter_value)
            q = port_q_direct[["series_key", "port", "year", "quarter", "log_LP"]].copy()
            q["n_obs"] = 1
            q["freq"] = "quarterly_direct"
            q["level"] = "port"
            q["terminal"] = ""
            out_rows.append(q)

    # ---------------------------------------------------------------------
    # Native quarterly terminal LP stays quarterly.
    # ---------------------------------------------------------------------
    # Same robustness idea: use the terminal keys plus the presence of a quarter field rather than
    # requiring freq == 'quarterly' literally.
    term_q = df[df["series_key"].isin(term_keys) & df["quarter"].notna()].copy()
    if not term_q.empty:
        term_q["quarter"] = term_q["quarter"].apply(parse_quarter_value)
        q = term_q[["series_key", "port", "terminal_canon", "year", "quarter", "log_LP"]].copy()
        q = q.rename(columns={"terminal_canon": "terminal"})
        q["n_obs"] = 1
        q["freq"] = "quarterly_native"
        q["level"] = "terminal"
        out_rows.append(q)

    if not out_rows:
        raise ValueError("No usable quarterly LP rows were constructed from LP_Panel.tsv")

    panel = pd.concat(out_rows, ignore_index=True, sort=False)
    panel["tq"] = [quarter_id(y, q) for y, q in zip(panel["year"], panel["quarter"])]
    panel["quarter_str"] = [quarter_label(y, q) for y, q in zip(panel["year"], panel["quarter"])]
    panel = panel.sort_values(["series_key", "tq"]).reset_index(drop=True)
    return panel


# ============================================================================
# Analysis-path construction
# ============================================================================

def add_competition_incumbent_paths(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the key analysis-side relabeling from the v4 architecture, but now at quarterly frequency.

    Haifa competition path:
      pre 2021Q3 -> Haifa port
      post 2021Q3 -> Haifa-Legacy terminal

    Ashdod competition path:
      pre 2022Q4 -> Ashdod port
      post 2022Q4 -> Ashdod-Legacy terminal

    This keeps the economic meaning from v4 while avoiding pseudo-monthly expansion.
    """
    rows = []

    # Keep the native terminal rows that are still needed directly.
    native_keep = panel[panel["series_key"].isin(["haifa_bayport", "haifa_legacy", "ashdod_legacy"])].copy()
    native_keep["analysis_unit"] = native_keep["series_key"]
    native_keep["analysis_port"] = native_keep["port"]
    native_keep["source_series"] = native_keep["series_key"]
    rows.append(native_keep)

    # Incumbent paths.
    haifa_comp_tq = quarter_id(2021, 3)
    ashdod_comp_tq = quarter_id(2022, 4)

    h_pre = panel[(panel["series_key"] == "haifa_port") & (panel["tq"] < haifa_comp_tq)].copy()
    h_post = panel[(panel["series_key"] == "haifa_legacy") & (panel["tq"] >= haifa_comp_tq)].copy()
    h_inc = pd.concat([h_pre, h_post], ignore_index=True)
    h_inc["analysis_unit"] = "haifa_incumbent"
    h_inc["analysis_port"] = "Haifa"
    h_inc["source_series"] = h_inc["series_key"]
    rows.append(h_inc)

    a_pre = panel[(panel["series_key"] == "ashdod_port") & (panel["tq"] < ashdod_comp_tq)].copy()
    a_post = panel[(panel["series_key"] == "ashdod_legacy") & (panel["tq"] >= ashdod_comp_tq)].copy()
    a_inc = pd.concat([a_pre, a_post], ignore_index=True)
    a_inc["analysis_unit"] = "ashdod_incumbent"
    a_inc["analysis_port"] = "Ashdod"
    a_inc["source_series"] = a_inc["series_key"]
    rows.append(a_inc)

    out = pd.concat(rows, ignore_index=True, sort=False)
    out = out.sort_values(["analysis_unit", "tq"]).reset_index(drop=True)
    return out


# ============================================================================
# Model design configuration
# ============================================================================

@dataclass(frozen=True)
class EventBin:
    label: str
    a: int
    b: int


@dataclass(frozen=True)
class ReformDesign:
    table_group: str
    reform: str
    target: str
    target_key: str
    control_units: Tuple[str, ...]
    reform_tq: int
    nyt_end_tq: Optional[int] = None


# Coarse quarterly bins.
# These are deliberately much broader than the old dense monthly path.
# The point is to make the dynamic design honest and estimable, not maximally high-dimensional.
EVENT_BINS: Tuple[EventBin, ...] = (
    EventBin("pre_8_5", -8, -5),
    EventBin("pre_4_2", -4, -2),
    EventBin("q0", 0, 0),
    EventBin("q1_2", 1, 2),
    EventBin("q3_4", 3, 4),
    EventBin("q5_8", 5, 8),
    EventBin("q9_40", 9, 40),
)

WINDOWS: Dict[str, Tuple[int, int]] = {
    # Pretrend window now defined in quarters, not months.
    "avg_pre": (-4, -2),
    # Conservative post windows start after the event quarter.
    "post_y1": (1, 4),
    "post_y1_2": (1, 8),
}


def build_reform_designs() -> List[ReformDesign]:
    haifa_comp_tq = quarter_id(2021, 3)
    ashdod_comp_tq = quarter_id(2022, 4)
    haifa_priv_tq = quarter_id(2023, 1)

    return [
        # Haifa competition: NYT object is genuinely local because Ashdod is not yet treated
        # until 2022Q4. For the benchmark/TWFE comparison we allow the full two-path sample.
        ReformDesign(
            table_group="competition",
            reform="haifa_comp",
            target="Haifa-Legacy",
            target_key="haifa_legacy",
            control_units=("ashdod_incumbent",),
            reform_tq=haifa_comp_tq,
            nyt_end_tq=quarter_id(2022, 3),
        ),
        # Ashdod competition remains benchmark/TWFE only in spirit. We still define the object here
        # so the script can run the conventional panel.
        ReformDesign(
            table_group="competition",
            reform="ashdod_comp",
            target="Ashdod-Legacy",
            target_key="ashdod_legacy",
            control_units=("haifa_incumbent",),
            reform_tq=ashdod_comp_tq,
            nyt_end_tq=None,
        ),
        # Haifa privatization: legacy treated, Bayport retained as comparison / placebo path.
        ReformDesign(
            table_group="privatization",
            reform="haifa_priv",
            target="Haifa-Legacy",
            target_key="haifa_legacy",
            control_units=("haifa_bayport",),
            reform_tq=haifa_priv_tq,
            nyt_end_tq=None,
        ),
        # Explicit placebo version retained because the thesis wants the comparison visible.
        ReformDesign(
            table_group="privatization",
            reform="haifa_priv",
            target="Haifa-Bayport",
            target_key="haifa_bayport_placebo",
            control_units=("haifa_legacy",),
            reform_tq=haifa_priv_tq,
            nyt_end_tq=None,
        ),
    ]


TARGET_TO_UNIT = {
    "Haifa-Legacy": "haifa_incumbent",  # competition object uses incumbent path
    "Ashdod-Legacy": "ashdod_incumbent",
    "Haifa-Bayport": "haifa_bayport",
    # For privatization legacy object we want native legacy terminal, not incumbent path.
    "Haifa-Legacy__priv": "haifa_legacy",
}


def target_unit_for_design(d: ReformDesign) -> str:
    if d.reform == "haifa_priv" and d.target == "Haifa-Legacy":
        return TARGET_TO_UNIT["Haifa-Legacy__priv"]
    return TARGET_TO_UNIT[d.target]


# ============================================================================
# Regression helpers
# ============================================================================

def event_bin_name(label: str) -> str:
    return f"bin_{label}"


def build_event_columns(df: pd.DataFrame, target_unit: str) -> pd.DataFrame:
    out = df.copy()
    out["treated_unit"] = (out["analysis_unit"] == target_unit).astype(int)
    for b in EVENT_BINS:
        col = event_bin_name(b.label)
        out[col] = ((out["treated_unit"] == 1) & (out["event_time"] >= b.a) & (out["event_time"] <= b.b)).astype(int)
    return out


def add_trend_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["t_centered"] = out["tq"] - out["tq"].mean()

    ports = sorted(out["analysis_port"].dropna().unique())
    if len(ports) <= 1:
        # Single-port sample (e.g. privatization). A common linear trend is all that is available.
        out["trend_common"] = out["t_centered"]
    else:
        for p in ports:
            out[f"trend_{norm_text(p)}"] = (out["analysis_port"] == p).astype(int) * out["t_centered"]
    return out


def build_formula(df: pd.DataFrame, dynamic_cols: Sequence[str], spec_name: str) -> str:
    rhs = ["C(analysis_unit)"]

    if spec_name == "baseline":
        # Main specification: quarter fixed effects.
        rhs.append("C(tq)")
    elif spec_name == "porttr":
        # Comparison / relaxed trend version.
        # We intentionally replace the fully nonparametric quarter FE shell with a linear trend shell.
        # This is NOT the preferred estimator philosophically; it is kept because the thesis wants the
        # comparison visible and because it can sometimes remain estimable when FE designs get thin.
        trend_cols = [c for c in df.columns if c.startswith("trend_")]
        rhs.extend(trend_cols)
    else:
        raise ValueError(f"Unknown spec_name={spec_name}")

    rhs.extend(dynamic_cols)
    return "log_LP ~ " + " + ".join(rhs)



def fit_ols(df: pd.DataFrame, formula: str):
    y, X = patsy.dmatrices(formula, data=df, return_type="dataframe")
    model = sm.OLS(y, X)
    res = model.fit(cov_type="HC1")
    return res, X



def weighted_average_from_bins(params: pd.Series, cov: pd.DataFrame, window: Tuple[int, int]) -> Tuple[float, float, List[str]]:
    """
    Compute a window average from coarse event bins.

    If a requested window only partially overlaps a bin, the coefficient receives weight equal to the
    number of quarters from that bin covered by the requested window.
    """
    a, b = window
    weights = {}
    for eb in EVENT_BINS:
        inter_a = max(a, eb.a)
        inter_b = min(b, eb.b)
        if inter_a <= inter_b:
            col = event_bin_name(eb.label)
            if col in params.index:
                weights[col] = inter_b - inter_a + 1

    if not weights:
        return np.nan, np.nan, []

    total_w = float(sum(weights.values()))
    w = np.array([weights[c] / total_w for c in weights], dtype=float)
    cols = list(weights.keys())
    beta = float(np.dot(w, params[cols]))
    subcov = cov.loc[cols, cols].to_numpy(dtype=float)
    var = float(w @ subcov @ w)
    se = math.sqrt(var) if var >= 0 else np.nan
    return beta, se, cols



def wald_zero_test(params: pd.Series, cov: pd.DataFrame, cols: Sequence[str]) -> Tuple[float, float, float, float]:
    """Wald test that selected coefficients are jointly zero using robust covariance."""
    if not cols:
        return np.nan, np.nan, np.nan, np.nan
    b = params[list(cols)].to_numpy(dtype=float)
    V = cov.loc[list(cols), list(cols)].to_numpy(dtype=float)
    try:
        Vinv = np.linalg.pinv(V)
        stat = float(b.T @ Vinv @ b)
        df_num = float(len(cols))
        # With HC-type inference there is no natural small-sample denominator df like the old FE F-test.
        df_denom = np.nan
        pvalue = float(1.0 - sm.stats.chisqprob(stat, int(df_num))) if hasattr(sm.stats, 'chisqprob') else float(sm.stats.stattools.stats.chisqprob(stat, int(df_num)))
    except Exception:
        # Fallback using scipy-like chi2 survival from statsmodels if available.
        from scipy.stats import chi2
        stat = float(b.T @ np.linalg.pinv(V) @ b)
        df_num = float(len(cols))
        df_denom = np.nan
        pvalue = float(chi2.sf(stat, df_num))
    return stat, pvalue, df_num, df_denom


# ============================================================================
# Sample construction
# ============================================================================

def subset_for_design(panel: pd.DataFrame, design: ReformDesign, design_type: str, max_pre: int = 8, max_post: int = 40) -> pd.DataFrame:
    target_unit = target_unit_for_design(design)
    keep_units = {target_unit, *design.control_units}
    df = panel[panel["analysis_unit"].isin(keep_units)].copy()

    # Haifa privatization requires native legacy + Bayport terminal comparison, not incumbent paths.
    if design.reform == "haifa_priv":
        if design.target == "Haifa-Legacy":
            keep_units = {"haifa_legacy", "haifa_bayport"}
        elif design.target == "Haifa-Bayport":
            keep_units = {"haifa_bayport", "haifa_legacy"}
        df = panel[panel["analysis_unit"].isin(keep_units)].copy()
        target_unit = target_unit_for_design(design)

    # Build local event time relative to THIS reform's calendar quarter.
    df["event_time"] = df["tq"] - design.reform_tq

    # Keep a bounded window to avoid very long tails that add nuisance dimensionality.
    df = df[(df["event_time"] >= -max_pre) & (df["event_time"] <= max_post)].copy()

    # For Haifa competition NYT, stop the sample before Ashdod becomes treated.
    if design_type == "NYT" and design.nyt_end_tq is not None:
        df = df[df["tq"] <= design.nyt_end_tq].copy()

    # Drop the omitted reference quarter (-1) from the dynamic shell only when we create event dummies;
    # keep it in the sample so the FE regression has the correct reference period.
    df = df.sort_values(["analysis_unit", "tq"]).reset_index(drop=True)
    return df


# ============================================================================
# Estimation
# ============================================================================

def run_dynamic_regression(df: pd.DataFrame, design: ReformDesign, design_type: str, spec_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_unit = target_unit_for_design(design)
    work = build_event_columns(df, target_unit)
    work = add_trend_columns(work)

    dynamic_cols = [event_bin_name(b.label) for b in EVENT_BINS]
    formula = build_formula(work, dynamic_cols, spec_name)
    res, X = fit_ols(work, formula)

    params = res.params
    cov = res.cov_params()

    dyn_rows = []
    for eb in EVENT_BINS:
        col = event_bin_name(eb.label)
        if col not in params.index:
            continue
        treated_obs_in_bin = int(((work["treated_unit"] == 1) & (work[col] == 1)).sum())
        dyn_rows.append(
            {
                "design": design_type,
                "table_group": design.table_group,
                "reform": design.reform,
                "target": design.target,
                "target_key": design.target_key,
                "spec_name": spec_name,
                "event_time": eb.a,
                "bin_label": eb.label,
                "a": eb.a,
                "b": eb.b,
                "j": eb.a,
                "beta": float(params[col]),
                "se": float(res.bse[col]),
                "t": float(res.tvalues[col]),
                "pvalue": float(res.pvalues[col]),
                "n_event_obs": treated_obs_in_bin,
                "n_obs": int(res.nobs),
                "r2": float(res.rsquared),
                "se_type": "HC1",
            }
        )

    # Coarse windows from the binned dynamic path.
    win_rows = []
    dynamic_param_names = [c for c in dynamic_cols if c in params.index]
    for window_name, ab in WINDOWS.items():
        beta, se, used_cols = weighted_average_from_bins(params, cov, ab)
        win_rows.append(
            {
                "design": design_type,
                "table_group": design.table_group,
                "reform": design.reform,
                "target": design.target,
                "target_key": design.target_key,
                "spec_name": spec_name,
                "window": window_name,
                "a": ab[0],
                "b": ab[1],
                "beta": beta,
                "se": se,
                "n_obs": int(res.nobs),
                "r2": float(res.rsquared),
                "used_bins": ",".join(used_cols),
            }
        )

    # Full-post window extends over all supported positive quarters after q0.
    post_cols = []
    for eb in EVENT_BINS:
        if eb.b < 1:
            continue
        col = event_bin_name(eb.label)
        if col in params.index:
            post_cols.append(col)
    if post_cols:
        # Use the same overlap-weight logic, but here with the effective positive range implied by bins.
        # We define the window as 1..max supported post quarter.
        max_post = max([b.b for b in EVENT_BINS if event_bin_name(b.label) in params.index and b.b >= 1])
        beta, se, used_cols = weighted_average_from_bins(params, cov, (1, max_post))
        win_rows.append(
            {
                "design": design_type,
                "table_group": design.table_group,
                "reform": design.reform,
                "target": design.target,
                "target_key": design.target_key,
                "spec_name": spec_name,
                "window": "full_post",
                "a": 1,
                "b": max_post,
                "beta": beta,
                "se": se,
                "n_obs": int(res.nobs),
                "r2": float(res.rsquared),
                "used_bins": ",".join(used_cols),
            }
        )

    # Joint pretrend test on the two coarse pre bins.
    pre_cols = [event_bin_name("pre_8_5"), event_bin_name("pre_4_2")]
    pre_cols = [c for c in pre_cols if c in params.index]
    stat, pvalue, df_num, df_denom = wald_zero_test(params, cov, pre_cols)
    pre_row = {
        "design": design_type,
        "table_group": design.table_group,
        "reform": design.reform,
        "target": design.target,
        "target_key": design.target_key,
        "spec_name": spec_name,
        "pre_min": -8,
        "pre_max": -2,
        "n_leads_total": 7,
        "n_bins_defined": 2,
        "n_bins_used": len(pre_cols),
        "f_stat": stat,
        "pvalue": pvalue,
        "df_num": df_num,
        "df_denom": df_denom,
        "n_obs": int(res.nobs),
        "r2": float(res.rsquared),
        "test_type": "chi2_wald_HC1",
    }

    return pd.DataFrame(dyn_rows), pd.DataFrame(win_rows), pd.DataFrame([pre_row])



def run_static_regression(df: pd.DataFrame, design: ReformDesign, spec_name: str) -> pd.DataFrame:
    """
    Static benchmark regressions.

    This remains in the pipeline because the thesis still wants the conventional comparison panel.
    The main change relative to v4 is that the static regressions are now run on the quarterly panel.
    """
    target_unit = target_unit_for_design(design)
    base = add_trend_columns(df.copy())
    base["treated_unit"] = (base["analysis_unit"] == target_unit).astype(int)

    rows = []
    horizons = {
        "full_post": (1, int(base["event_time"].max())),
        "post_y1": (1, 4),
        "post_y1_2": (1, 8),
    }

    for horizon_name, (a, b) in horizons.items():
        work = base.copy()
        work = work[(work["event_time"] >= -8) & (work["event_time"] <= b)].copy()
        work["treated_post"] = ((work["treated_unit"] == 1) & (work["event_time"] >= a) & (work["event_time"] <= b)).astype(int)

        rhs = ["C(analysis_unit)"]
        if spec_name == "baseline":
            rhs.append("C(tq)")
        elif spec_name == "porttr":
            rhs.extend([c for c in work.columns if c.startswith("trend_")])
        else:
            raise ValueError(f"Unknown spec_name={spec_name}")
        rhs.append("treated_post")
        formula = "log_LP ~ " + " + ".join(rhs)

        res, X = fit_ols(work, formula)
        if "treated_post" not in res.params.index:
            continue

        post_treated = int(((work["treated_unit"] == 1) & (work["event_time"] >= a) & (work["event_time"] <= b)).sum())
        rows.append(
            {
                "design": "TWFE",
                "table_group": design.table_group,
                "reform": design.reform,
                "target": design.target,
                "target_key": design.target_key,
                "spec_name": spec_name,
                "horizon": horizon_name,
                "a": a,
                "b": b,
                "beta": float(res.params["treated_post"]),
                "se": float(res.bse["treated_post"]),
                "pvalue": float(res.pvalues["treated_post"]),
                "n_obs": int(res.nobs),
                "r2": float(res.rsquared),
                "n_treated": int((work["treated_unit"] == 1).sum()),
                "n_control": int((work["treated_unit"] == 0).sum()),
                "n_post_treated": post_treated,
                "se_type": "HC1",
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# Main orchestration
# ============================================================================

def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False)



def main() -> None:
    parser = argparse.ArgumentParser(description="Quarterly rewrite of Model 1A")
    parser.add_argument("--lp", default=None, help="Optional path to mixed-frequency LP_Panel.tsv; if omitted, the script prefers the common_rule_v5 LP build and then falls back to the canonical LP panel.")
    parser.add_argument("--out", default="Design/Output (new)/Model_1A_q", help="Output directory")
    args = parser.parse_args()

    lp_path = resolve_lp_path(args.lp)
    print(f"[Model_1A_v5_quarterly] Using LP panel: {lp_path}")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1. Honest quarterly panel construction.
    # ------------------------------------------------------------------
    q_panel = build_quarterly_lp_panel(lp_path)
    q_panel = add_competition_incumbent_paths(q_panel)

    # Keep only the analysis units we actually use in this version.
    q_panel = q_panel[q_panel["analysis_unit"].isin([
        "haifa_incumbent", "ashdod_incumbent", "haifa_legacy", "haifa_bayport"
    ])].copy()

    write_tsv(q_panel, outdir / "model1a_q_analysis_panel.tsv")

    # ------------------------------------------------------------------
    # Step 2. Run dynamic NYT and TWFE-style benchmark objects.
    # ------------------------------------------------------------------
    dynamic_nyt = []
    dynamic_twfe = []
    windows_nyt = []
    windows_twfe = []
    pretrend_nyt = []
    pretrend_twfe = []
    static_twfe = []

    designs = build_reform_designs()

    for d in designs:
        # NYT objects only where conceptually intended.
        if d.reform != "ashdod_comp":
            for spec_name in ("baseline", "porttr"):
                sample = subset_for_design(q_panel, d, design_type="NYT")
                # For privatization keep only periods where both legacy and bayport exist.
                if d.reform == "haifa_priv":
                    sample = sample[sample["tq"] >= quarter_id(2021, 3)].copy()
                if sample.empty:
                    continue
                dyn, win, pre = run_dynamic_regression(sample, d, design_type="NYT", spec_name=spec_name)
                dynamic_nyt.append(dyn)
                windows_nyt.append(win)
                pretrend_nyt.append(pre)

        # Conventional benchmark panel retained because the thesis still wants that comparison visible.
        for spec_name in ("baseline", "porttr"):
            sample = subset_for_design(q_panel, d, design_type="TWFE")
            if d.reform == "haifa_priv":
                sample = sample[sample["tq"] >= quarter_id(2021, 3)].copy()
            if sample.empty:
                continue
            dyn, win, pre = run_dynamic_regression(sample, d, design_type="TWFE", spec_name=spec_name)
            dynamic_twfe.append(dyn)
            windows_twfe.append(win)
            pretrend_twfe.append(pre)
            static_rows = run_static_regression(sample, d, spec_name=spec_name)
            static_twfe.append(static_rows)

    # ------------------------------------------------------------------
    # Step 3. Write pooled outputs.
    # ------------------------------------------------------------------
    dynamic_nyt_df = pd.concat(dynamic_nyt, ignore_index=True) if dynamic_nyt else pd.DataFrame()
    dynamic_twfe_df = pd.concat(dynamic_twfe, ignore_index=True) if dynamic_twfe else pd.DataFrame()
    windows_nyt_df = pd.concat(windows_nyt, ignore_index=True) if windows_nyt else pd.DataFrame()
    windows_twfe_df = pd.concat(windows_twfe, ignore_index=True) if windows_twfe else pd.DataFrame()
    pretrend_nyt_df = pd.concat(pretrend_nyt, ignore_index=True) if pretrend_nyt else pd.DataFrame()
    pretrend_twfe_df = pd.concat(pretrend_twfe, ignore_index=True) if pretrend_twfe else pd.DataFrame()
    static_twfe_df = pd.concat(static_twfe, ignore_index=True) if static_twfe else pd.DataFrame()

    write_tsv(dynamic_nyt_df, outdir / "model1a_q_lp_dynamic_betas_all.tsv")
    write_tsv(dynamic_twfe_df, outdir / "model1a_q_lp_dynamic_betas_all_twfe.tsv")
    write_tsv(windows_nyt_df, outdir / "model1a_q_lp_window_betas_all.tsv")
    write_tsv(windows_twfe_df, outdir / "model1a_q_lp_window_betas_all_twfe.tsv")
    write_tsv(pretrend_nyt_df, outdir / "model1a_q_lp_pretrend_tests_all.tsv")
    write_tsv(pretrend_twfe_df, outdir / "model1a_q_lp_pretrend_tests_all_twfe.tsv")
    write_tsv(static_twfe_df, outdir / "model1a_q_lp_static_betas_all_twfe.tsv")

    manifest = {
        "script": "Model_1A_v5_quarterly.py",
        "main_changes_vs_v4": [
            "Uses LP_Panel.tsv rather than LP_Panel_monthly.tsv",
            "Aggregates pre-reform monthly port LP to quarters using mean(log(LP))",
            "Keeps post-reform terminal LP at native quarterly frequency",
            "Estimates quarterly event-study objects instead of pseudo-monthly event studies",
            "Uses coarse quarterly event-time bins to reduce saturation",
            "Retains NYT and conventional benchmark panels",
            "Retains a trend-based comparison spec ('porttr') but treats it as a benchmark, not a replacement for the main design",
            "Uses HC1 standard errors by default because unit-level clustering is unreliable in the tiny two-path samples",
        ],
        "event_bins": [eb.__dict__ for eb in EVENT_BINS],
        "windows": WINDOWS,
        "outputs": [
            "model1a_q_lp_dynamic_betas_all.tsv",
            "model1a_q_lp_dynamic_betas_all_twfe.tsv",
            "model1a_q_lp_window_betas_all.tsv",
            "model1a_q_lp_window_betas_all_twfe.tsv",
            "model1a_q_lp_pretrend_tests_all.tsv",
            "model1a_q_lp_pretrend_tests_all_twfe.tsv",
            "model1a_q_lp_static_betas_all_twfe.tsv",
            "model1a_q_analysis_panel.tsv",
        ],
    }
    with open(outdir / "model1a_q_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Wrote quarterly Model 1A outputs to:", outdir)


if __name__ == "__main__":
    main()
