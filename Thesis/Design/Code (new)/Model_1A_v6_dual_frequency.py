#!/usr/bin/env python3
"""
Model_1A_v6_dual_frequency.py

What this file does
-------------------
This is a dual-frequency rewrite of Model 1A that allows the thesis workflow to run
both a monthly shell and an honest quarterly shell from the same mixed-frequency LP panel.
It is meant for comparison and diagnosis, not just replacement.

Main goals
----------
1. Fix the quarter parsing bug that broke Model_1A_v5_quarterly.py when quarter values are
   stored like 'Q3' rather than 3.
2. Let the user run:
      - monthly only
      - quarterly only
      - both monthly and quarterly
3. Allow the user to point the model at whichever LP proxy branch they want, including:
      - Data/LP/common_rule_v5/LP_Panel.tsv
      - Data/LP/raw_from_l_v2/LP_Panel.tsv
      - Data/LP/LP_Panel.tsv
4. Produce side-by-side comparable output families for monthly and quarterly results.

Important design note
---------------------
The monthly branch expands quarterly terminal observations into within-quarter monthly step
functions internally, analogous to preprocess_LP_Panel.py. This is useful for direct comparison
against the old monthly event-study shell, but it is not the preferred causal object when the
terminal LP information is genuinely quarterly.

The quarterly branch keeps post-reform terminal LP at native quarterly frequency and collapses
pre-reform monthly port LP to quarters by averaging log(LP) within each quarter.
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
import patsy
import statsmodels.api as sm


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = s.replace("—", "-").replace("–", "-").replace("_", "-")
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def parse_quarter_value(x) -> int:
    if pd.isna(x):
        raise ValueError("Quarter value is missing")
    s = str(x).strip()
    if s.upper().startswith("Q"):
        s = s[1:]
    return int(float(s))


def quarter_from_month(month: int) -> int:
    return ((int(month) - 1) // 3) + 1


def quarter_id(year: int, quarter: int) -> int:
    return int(year) * 4 + int(quarter) - 1


def month_id(year: int, month: int) -> int:
    return int(year) * 12 + int(month)


def quarter_label(year: int, quarter: int) -> str:
    return f"{int(year)}Q{int(quarter)}"


def month_label(year: int, month: int) -> str:
    return f"{int(year)}-{int(month):02d}"


def months_in_quarter(year: int, quarter: int) -> List[Tuple[int, int]]:
    q = int(quarter)
    start_m = 3 * (q - 1) + 1
    return [(int(year), start_m + i) for i in range(3)]


# -----------------------------------------------------------------------------
# Canonical naming
# -----------------------------------------------------------------------------

def canonical_terminal(port: str, terminal: str, series_id: str) -> str:
    p = norm_text(port)
    t = norm_text(terminal)
    sid = norm_text(series_id)

    if "haifa" in sid and ("bay" in sid or "sipg" in sid):
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

    if term == "Haifa-Legacy":
        return "haifa_legacy"
    if term == "Haifa-Bayport":
        return "haifa_bayport"
    if term == "Ashdod-Legacy":
        return "ashdod_legacy"
    if term == "Ashdod-HCT":
        return "ashdod_hct"
    return None


# -----------------------------------------------------------------------------
# Input resolution
# -----------------------------------------------------------------------------

def resolve_lp_path(user_path: Optional[str]) -> Path:
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"Requested LP panel does not exist: {p}")
        return p
    candidates = [
        Path("Data/LP/common_rule_v5/LP_Panel.tsv"),
        Path("Data/LP/raw_from_l_v2/LP_Panel.tsv"),
        Path("Data/LP/LP_Panel.tsv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not resolve an LP_Panel.tsv path")


def load_lp_panel(lp_path: Path) -> pd.DataFrame:
    df = pd.read_csv(lp_path, sep="\t")
    required = {"port", "level", "series_id", "LP", "freq", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"LP panel missing required columns: {sorted(missing)}")
    if "terminal" not in df.columns:
        df["terminal"] = ""
    if "quarter" not in df.columns:
        df["quarter"] = np.nan
    if "month" not in df.columns:
        df["month"] = np.nan
    df = df.copy()
    df = df[df["LP"].notna()].copy()
    df = df[df["LP"] > 0].copy()
    df["terminal_canon"] = [canonical_terminal(p, t, s) for p, t, s in zip(df["port"], df["terminal"], df["series_id"])]
    df["series_key"] = [canonical_series_key(p, l, t, s) for p, l, t, s in zip(df["port"], df["level"], df["terminal"], df["series_id"])]
    df = df[df["series_key"].notna()].copy()
    df["log_LP"] = np.log(df["LP"].astype(float))
    return df


# -----------------------------------------------------------------------------
# Panel builders
# -----------------------------------------------------------------------------

def build_quarterly_panel(df: pd.DataFrame) -> pd.DataFrame:
    out_rows: List[pd.DataFrame] = []
    port_keys = {"haifa_port", "ashdod_port"}
    term_keys = {"haifa_legacy", "haifa_bayport", "ashdod_legacy", "ashdod_hct"}

    port_month = df[df["series_key"].isin(port_keys) & df["month"].notna()].copy()
    if not port_month.empty:
        port_month["month"] = pd.to_numeric(port_month["month"], errors="coerce")
        port_month = port_month[port_month["month"].notna()].copy()
        port_month["month"] = port_month["month"].astype(int)
        port_month["quarter_num"] = port_month["month"].apply(quarter_from_month)
        q = (
            port_month.groupby(["series_key", "port", "year", "quarter_num"], as_index=False)
            .agg(log_LP=("log_LP", "mean"), n_obs=("log_LP", "size"))
        )
        q["time_id"] = [quarter_id(y, qn) for y, qn in zip(q["year"], q["quarter_num"])]
        q["time_str"] = [quarter_label(y, qn) for y, qn in zip(q["year"], q["quarter_num"])]
        q["terminal"] = ""
        q["analysis_freq"] = "quarterly"
        out_rows.append(q[["series_key", "port", "terminal", "year", "quarter_num", "time_id", "time_str", "log_LP", "n_obs", "analysis_freq"]])

    term_q = df[df["series_key"].isin(term_keys) & df["quarter"].notna()].copy()
    if not term_q.empty:
        term_q["quarter_num"] = term_q["quarter"].apply(parse_quarter_value)
        q = term_q[["series_key", "port", "terminal_canon", "year", "quarter_num", "log_LP"]].copy()
        q = q.rename(columns={"terminal_canon": "terminal"})
        q["n_obs"] = 1
        q["time_id"] = [quarter_id(y, qn) for y, qn in zip(q["year"], q["quarter_num"])]
        q["time_str"] = [quarter_label(y, qn) for y, qn in zip(q["year"], q["quarter_num"])]
        q["analysis_freq"] = "quarterly"
        out_rows.append(q[["series_key", "port", "terminal", "year", "quarter_num", "time_id", "time_str", "log_LP", "n_obs", "analysis_freq"]])

    if not out_rows:
        raise ValueError("No quarterly rows could be constructed from LP panel")

    panel = pd.concat(out_rows, ignore_index=True, sort=False)
    panel["month_num"] = np.nan
    panel = panel.sort_values(["series_key", "time_id"]).reset_index(drop=True)
    return panel


def build_monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    out_rows: List[pd.DataFrame] = []
    port_keys = {"haifa_port", "ashdod_port"}
    term_keys = {"haifa_legacy", "haifa_bayport", "ashdod_legacy", "ashdod_hct"}

    port_month = df[df["series_key"].isin(port_keys) & df["month"].notna()].copy()
    if not port_month.empty:
        port_month["month_num"] = pd.to_numeric(port_month["month"], errors="coerce")
        port_month = port_month[port_month["month_num"].notna()].copy()
        port_month["month_num"] = port_month["month_num"].astype(int)
        m = port_month[["series_key", "port", "year", "month_num", "log_LP"]].copy()
        m["quarter_num"] = m["month_num"].apply(quarter_from_month)
        m["time_id"] = [month_id(y, mn) for y, mn in zip(m["year"], m["month_num"])]
        m["time_str"] = [month_label(y, mn) for y, mn in zip(m["year"], m["month_num"])]
        m["terminal"] = ""
        m["n_obs"] = 1
        m["analysis_freq"] = "monthly"
        out_rows.append(m[["series_key", "port", "terminal", "year", "month_num", "quarter_num", "time_id", "time_str", "log_LP", "n_obs", "analysis_freq"]])

    term_q = df[df["series_key"].isin(term_keys) & df["quarter"].notna()].copy()
    if not term_q.empty:
        term_q["quarter_num"] = term_q["quarter"].apply(parse_quarter_value)
        expanded_rows = []
        for _, r in term_q.iterrows():
            months = months_in_quarter(int(r["year"]), int(r["quarter_num"]))
            for y, m in months:
                expanded_rows.append({
                    "series_key": r["series_key"],
                    "port": r["port"],
                    "terminal": r["terminal_canon"],
                    "year": int(y),
                    "month_num": int(m),
                    "quarter_num": int(r["quarter_num"]),
                    "time_id": month_id(y, m),
                    "time_str": month_label(y, m),
                    "log_LP": float(r["log_LP"]),
                    "n_obs": 1,
                    "analysis_freq": "monthly",
                })
        out_rows.append(pd.DataFrame(expanded_rows))

    if not out_rows:
        raise ValueError("No monthly rows could be constructed from LP panel")

    panel = pd.concat(out_rows, ignore_index=True, sort=False)
    panel = panel.sort_values(["series_key", "time_id"]).reset_index(drop=True)
    return panel


# -----------------------------------------------------------------------------
# Analysis-path construction
# -----------------------------------------------------------------------------

def add_analysis_paths(panel: pd.DataFrame, freq: str) -> pd.DataFrame:
    rows = []
    native_keep = panel[panel["series_key"].isin(["haifa_bayport", "haifa_legacy", "ashdod_legacy"])].copy()
    native_keep["analysis_unit"] = native_keep["series_key"]
    native_keep["analysis_port"] = native_keep["port"]
    native_keep["source_series"] = native_keep["series_key"]
    rows.append(native_keep)

    if freq == "quarterly":
        haifa_comp_time = quarter_id(2021, 3)
        ashdod_comp_time = quarter_id(2022, 4)
    else:
        haifa_comp_time = month_id(2021, 9)
        ashdod_comp_time = month_id(2022, 11)

    h_pre = panel[(panel["series_key"] == "haifa_port") & (panel["time_id"] < haifa_comp_time)].copy()
    h_post = panel[(panel["series_key"] == "haifa_legacy") & (panel["time_id"] >= haifa_comp_time)].copy()
    h_inc = pd.concat([h_pre, h_post], ignore_index=True)
    h_inc["analysis_unit"] = "haifa_incumbent"
    h_inc["analysis_port"] = "Haifa"
    h_inc["source_series"] = h_inc["series_key"]
    rows.append(h_inc)

    a_pre = panel[(panel["series_key"] == "ashdod_port") & (panel["time_id"] < ashdod_comp_time)].copy()
    a_post = panel[(panel["series_key"] == "ashdod_legacy") & (panel["time_id"] >= ashdod_comp_time)].copy()
    a_inc = pd.concat([a_pre, a_post], ignore_index=True)
    a_inc["analysis_unit"] = "ashdod_incumbent"
    a_inc["analysis_port"] = "Ashdod"
    a_inc["source_series"] = a_inc["series_key"]
    rows.append(a_inc)

    out = pd.concat(rows, ignore_index=True, sort=False)
    out = out.sort_values(["analysis_unit", "time_id"]).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
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
    reform_time: int
    nyt_end_time: Optional[int] = None


MONTHLY_BINS: Tuple[EventBin, ...] = (
    EventBin("pre_24_13", -24, -13),
    EventBin("pre_12_2", -12, -2),
    EventBin("m0", 0, 0),
    EventBin("m1_3", 1, 3),
    EventBin("m4_6", 4, 6),
    EventBin("m7_12", 7, 12),
    EventBin("m13_24", 13, 24),
    EventBin("m25_120", 25, 120),
)

MONTHLY_WINDOWS: Dict[str, Tuple[int, int]] = {
    "avg_pre": (-12, -2),
    "post_y1": (1, 12),
    "post_y1_2": (1, 24),
}

QUARTERLY_BINS: Tuple[EventBin, ...] = (
    EventBin("pre_8_5", -8, -5),
    EventBin("pre_4_2", -4, -2),
    EventBin("q0", 0, 0),
    EventBin("q1_2", 1, 2),
    EventBin("q3_4", 3, 4),
    EventBin("q5_8", 5, 8),
    EventBin("q9_40", 9, 40),
)

QUARTERLY_WINDOWS: Dict[str, Tuple[int, int]] = {
    "avg_pre": (-4, -2),
    "post_y1": (1, 4),
    "post_y1_2": (1, 8),
}


def build_designs(freq: str) -> List[ReformDesign]:
    if freq == "quarterly":
        haifa_comp = quarter_id(2021, 3)
        ashdod_comp = quarter_id(2022, 4)
        haifa_priv = quarter_id(2023, 1)
        haifa_nyt_end = quarter_id(2022, 3)
    else:
        haifa_comp = month_id(2021, 9)
        ashdod_comp = month_id(2022, 11)
        haifa_priv = month_id(2023, 1)
        haifa_nyt_end = month_id(2022, 10)
    return [
        ReformDesign("competition", "haifa_comp", "Haifa-Legacy", "haifa_legacy", ("ashdod_incumbent",), haifa_comp, haifa_nyt_end),
        ReformDesign("competition", "ashdod_comp", "Ashdod-Legacy", "ashdod_legacy", ("haifa_incumbent",), ashdod_comp, None),
        ReformDesign("privatization", "haifa_priv", "Haifa-Legacy", "haifa_legacy", ("haifa_bayport",), haifa_priv, None),
        ReformDesign("privatization", "haifa_priv", "Haifa-Bayport", "haifa_bayport_placebo", ("haifa_legacy",), haifa_priv, None),
    ]


TARGET_TO_UNIT = {
    "Haifa-Legacy": "haifa_incumbent",
    "Ashdod-Legacy": "ashdod_incumbent",
    "Haifa-Bayport": "haifa_bayport",
    "Haifa-Legacy__priv": "haifa_legacy",
}


def target_unit_for_design(d: ReformDesign) -> str:
    if d.reform == "haifa_priv" and d.target == "Haifa-Legacy":
        return TARGET_TO_UNIT["Haifa-Legacy__priv"]
    return TARGET_TO_UNIT[d.target]


# -----------------------------------------------------------------------------
# Regression helpers
# -----------------------------------------------------------------------------

def event_bin_name(label: str) -> str:
    return f"bin_{label}"


def build_event_columns(df: pd.DataFrame, target_unit: str, bins: Sequence[EventBin]) -> pd.DataFrame:
    out = df.copy()
    out["treated_unit"] = (out["analysis_unit"] == target_unit).astype(int)
    for b in bins:
        col = event_bin_name(b.label)
        out[col] = ((out["treated_unit"] == 1) & (out["event_time"] >= b.a) & (out["event_time"] <= b.b)).astype(int)
    return out


def add_trend_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["t_centered"] = out["time_id"] - out["time_id"].mean()
    ports = sorted(out["analysis_port"].dropna().unique())
    if len(ports) <= 1:
        out["trend_common"] = out["t_centered"]
    else:
        for p in ports:
            out[f"trend_{norm_text(p)}"] = (out["analysis_port"] == p).astype(int) * out["t_centered"]
    return out


def build_formula(df: pd.DataFrame, dynamic_cols: Sequence[str], spec_name: str) -> str:
    rhs = ["C(analysis_unit)"]
    if spec_name == "baseline":
        rhs.append("C(time_id)")
    elif spec_name == "porttr":
        rhs.extend([c for c in df.columns if c.startswith("trend_")])
    else:
        raise ValueError(f"Unknown spec_name={spec_name}")
    rhs.extend(dynamic_cols)
    return "log_LP ~ " + " + ".join(rhs)


def fit_ols(df: pd.DataFrame, formula: str):
    y, X = patsy.dmatrices(formula, data=df, return_type="dataframe")
    model = sm.OLS(y, X)
    res = model.fit(cov_type="HC1")
    return res


def weighted_average_from_bins(params: pd.Series, cov: pd.DataFrame, window: Tuple[int, int], bins: Sequence[EventBin]) -> Tuple[float, float, List[str]]:
    a, b = window
    weights = {}
    for eb in bins:
        inter_a = max(a, eb.a)
        inter_b = min(b, eb.b)
        if inter_a <= inter_b:
            col = event_bin_name(eb.label)
            if col in params.index:
                weights[col] = inter_b - inter_a + 1
    if not weights:
        return np.nan, np.nan, []
    total_w = float(sum(weights.values()))
    cols = list(weights.keys())
    w = np.array([weights[c] / total_w for c in cols], dtype=float)
    beta = float(np.dot(w, params[cols]))
    subcov = cov.loc[cols, cols].to_numpy(dtype=float)
    var = float(w @ subcov @ w)
    se = math.sqrt(var) if var >= 0 else np.nan
    return beta, se, cols


def wald_zero_test(params: pd.Series, cov: pd.DataFrame, cols: Sequence[str]) -> Tuple[float, float, float, float]:
    if not cols:
        return np.nan, np.nan, np.nan, np.nan
    b = params[list(cols)].to_numpy(dtype=float)
    V = cov.loc[list(cols), list(cols)].to_numpy(dtype=float)
    try:
        Vinv = np.linalg.pinv(V)
        stat = float(b.T @ Vinv @ b)
        df_num = float(len(cols))
        from scipy.stats import chi2
        pvalue = float(chi2.sf(stat, df_num))
        return stat, pvalue, df_num, np.nan
    except Exception:
        return np.nan, np.nan, float(len(cols)), np.nan


# -----------------------------------------------------------------------------
# Sample construction
# -----------------------------------------------------------------------------

def subset_for_design(panel: pd.DataFrame, design: ReformDesign, design_type: str, max_pre: int, max_post: int, freq: str) -> pd.DataFrame:
    target_unit = target_unit_for_design(design)
    keep_units = {target_unit, *design.control_units}
    df = panel[panel["analysis_unit"].isin(keep_units)].copy()

    if design.reform == "haifa_priv":
        keep_units = {"haifa_legacy", "haifa_bayport"}
        df = panel[panel["analysis_unit"].isin(keep_units)].copy()
        target_unit = target_unit_for_design(design)

    df["event_time"] = df["time_id"] - design.reform_time
    df = df[(df["event_time"] >= -max_pre) & (df["event_time"] <= max_post)].copy()

    if design_type == "NYT" and design.nyt_end_time is not None:
        df = df[df["time_id"] <= design.nyt_end_time].copy()

    if design.reform == "haifa_priv":
        start = quarter_id(2021, 3) if freq == "quarterly" else month_id(2021, 9)
        df = df[df["time_id"] >= start].copy()

    df = df.sort_values(["analysis_unit", "time_id"]).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Estimation
# -----------------------------------------------------------------------------

def run_dynamic_regression(df: pd.DataFrame, design: ReformDesign, design_type: str, spec_name: str, bins: Sequence[EventBin], windows: Dict[str, Tuple[int, int]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_unit = target_unit_for_design(design)
    work = build_event_columns(df, target_unit, bins)
    work = add_trend_columns(work)
    dynamic_cols = [event_bin_name(b.label) for b in bins]
    formula = build_formula(work, dynamic_cols, spec_name)
    res = fit_ols(work, formula)
    params = res.params
    cov = res.cov_params()

    dyn_rows = []
    for eb in bins:
        col = event_bin_name(eb.label)
        if col not in params.index:
            continue
        treated_obs_in_bin = int(((work["treated_unit"] == 1) & (work[col] == 1)).sum())
        dyn_rows.append({
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
        })

    win_rows = []
    for window_name, ab in windows.items():
        beta, se, used_cols = weighted_average_from_bins(params, cov, ab, bins)
        win_rows.append({
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
        })

    post_cols = []
    for eb in bins:
        if eb.b < 1:
            continue
        col = event_bin_name(eb.label)
        if col in params.index:
            post_cols.append(col)
    if post_cols:
        max_post = max([b.b for b in bins if event_bin_name(b.label) in params.index and b.b >= 1])
        beta, se, used_cols = weighted_average_from_bins(params, cov, (1, max_post), bins)
        win_rows.append({
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
        })

    pre_bins = [b for b in bins if b.b <= -2]
    pre_cols = [event_bin_name(b.label) for b in pre_bins if event_bin_name(b.label) in params.index]
    stat, pvalue, df_num, df_denom = wald_zero_test(params, cov, pre_cols)
    pre_row = {
        "design": design_type,
        "table_group": design.table_group,
        "reform": design.reform,
        "target": design.target,
        "target_key": design.target_key,
        "spec_name": spec_name,
        "pre_min": float(min([b.a for b in pre_bins])) if pre_bins else np.nan,
        "pre_max": float(max([b.b for b in pre_bins])) if pre_bins else np.nan,
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


def run_static_regression(df: pd.DataFrame, design: ReformDesign, spec_name: str, windows: Dict[str, Tuple[int, int]]) -> pd.DataFrame:
    target_unit = target_unit_for_design(design)
    base = add_trend_columns(df.copy())
    base["treated_unit"] = (base["analysis_unit"] == target_unit).astype(int)
    rows = []
    horizons = {
        "full_post": (1, int(base["event_time"].max())),
        "post_y1": windows["post_y1"],
        "post_y1_2": windows["post_y1_2"],
    }
    for horizon_name, (a, b) in horizons.items():
        work = base.copy()
        work = work[(work["event_time"] >= min(-24, int(base["event_time"].min()))) & (work["event_time"] <= b)].copy()
        work["treated_post"] = ((work["treated_unit"] == 1) & (work["event_time"] >= a) & (work["event_time"] <= b)).astype(int)
        rhs = ["C(analysis_unit)"]
        if spec_name == "baseline":
            rhs.append("C(time_id)")
        else:
            rhs.extend([c for c in work.columns if c.startswith("trend_")])
        rhs.append("treated_post")
        formula = "log_LP ~ " + " + ".join(rhs)
        res = fit_ols(work, formula)
        if "treated_post" not in res.params.index:
            continue
        rows.append({
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
            "n_post_treated": int(((work["treated_unit"] == 1) & (work["event_time"] >= a) & (work["event_time"] <= b)).sum()),
            "se_type": "HC1",
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Frequency runner
# -----------------------------------------------------------------------------

def run_frequency(panel: pd.DataFrame, freq: str, outdir: Path) -> Dict[str, pd.DataFrame]:
    if freq == "quarterly":
        bins = QUARTERLY_BINS
        windows = QUARTERLY_WINDOWS
        max_pre, max_post = 8, 40
        suffix = "q"
    else:
        bins = MONTHLY_BINS
        windows = MONTHLY_WINDOWS
        max_pre, max_post = 24, 120
        suffix = "m"

    q_panel = add_analysis_paths(panel, freq)
    q_panel = q_panel[q_panel["analysis_unit"].isin(["haifa_incumbent", "ashdod_incumbent", "haifa_legacy", "haifa_bayport"])].copy()
    q_panel.to_csv(outdir / f"model1a_{suffix}_analysis_panel.tsv", sep="\t", index=False)

    dynamic_nyt = []
    dynamic_twfe = []
    windows_nyt = []
    windows_twfe = []
    pretrend_nyt = []
    pretrend_twfe = []
    static_twfe = []

    for d in build_designs(freq):
        if d.reform != "ashdod_comp":
            for spec_name in ("baseline", "porttr"):
                sample = subset_for_design(q_panel, d, "NYT", max_pre, max_post, freq)
                if sample.empty:
                    continue
                dyn, win, pre = run_dynamic_regression(sample, d, "NYT", spec_name, bins, windows)
                dynamic_nyt.append(dyn)
                windows_nyt.append(win)
                pretrend_nyt.append(pre)

        for spec_name in ("baseline", "porttr"):
            sample = subset_for_design(q_panel, d, "TWFE", max_pre, max_post, freq)
            if sample.empty:
                continue
            dyn, win, pre = run_dynamic_regression(sample, d, "TWFE", spec_name, bins, windows)
            dynamic_twfe.append(dyn)
            windows_twfe.append(win)
            pretrend_twfe.append(pre)
            static_rows = run_static_regression(sample, d, spec_name, windows)
            static_twfe.append(static_rows)

    out = {
        "dynamic_nyt": pd.concat(dynamic_nyt, ignore_index=True) if dynamic_nyt else pd.DataFrame(),
        "dynamic_twfe": pd.concat(dynamic_twfe, ignore_index=True) if dynamic_twfe else pd.DataFrame(),
        "windows_nyt": pd.concat(windows_nyt, ignore_index=True) if windows_nyt else pd.DataFrame(),
        "windows_twfe": pd.concat(windows_twfe, ignore_index=True) if windows_twfe else pd.DataFrame(),
        "pretrend_nyt": pd.concat(pretrend_nyt, ignore_index=True) if pretrend_nyt else pd.DataFrame(),
        "pretrend_twfe": pd.concat(pretrend_twfe, ignore_index=True) if pretrend_twfe else pd.DataFrame(),
        "static_twfe": pd.concat(static_twfe, ignore_index=True) if static_twfe else pd.DataFrame(),
    }

    out["dynamic_nyt"].to_csv(outdir / f"model1a_{suffix}_lp_dynamic_betas_all.tsv", sep="\t", index=False)
    out["dynamic_twfe"].to_csv(outdir / f"model1a_{suffix}_lp_dynamic_betas_all_twfe.tsv", sep="\t", index=False)
    out["windows_nyt"].to_csv(outdir / f"model1a_{suffix}_lp_window_betas_all.tsv", sep="\t", index=False)
    out["windows_twfe"].to_csv(outdir / f"model1a_{suffix}_lp_window_betas_all_twfe.tsv", sep="\t", index=False)
    out["pretrend_nyt"].to_csv(outdir / f"model1a_{suffix}_lp_pretrend_tests_all.tsv", sep="\t", index=False)
    out["pretrend_twfe"].to_csv(outdir / f"model1a_{suffix}_lp_pretrend_tests_all_twfe.tsv", sep="\t", index=False)
    out["static_twfe"].to_csv(outdir / f"model1a_{suffix}_lp_static_betas_all_twfe.tsv", sep="\t", index=False)
    return out


def build_comparisons(monthly: Dict[str, pd.DataFrame], quarterly: Dict[str, pd.DataFrame], outdir: Path) -> None:
    key_win = ["design", "table_group", "reform", "target", "target_key", "spec_name", "window"]
    if not monthly["windows_nyt"].empty or not quarterly["windows_nyt"].empty:
        mwin = pd.concat([monthly["windows_nyt"], monthly["windows_twfe"]], ignore_index=True)
        qwin = pd.concat([quarterly["windows_nyt"], quarterly["windows_twfe"]], ignore_index=True)
        keep_cols = key_win + ["beta", "se", "n_obs", "r2"]
        comp = mwin[keep_cols].merge(qwin[keep_cols], on=key_win, how="outer", suffixes=("_m", "_q"))
        comp.to_csv(outdir / "model1a_mq_window_compare.tsv", sep="\t", index=False)

    key_pre = ["design", "table_group", "reform", "target", "target_key", "spec_name"]
    if not monthly["pretrend_nyt"].empty or not quarterly["pretrend_nyt"].empty:
        mpre = pd.concat([monthly["pretrend_nyt"], monthly["pretrend_twfe"]], ignore_index=True)
        qpre = pd.concat([quarterly["pretrend_nyt"], quarterly["pretrend_twfe"]], ignore_index=True)
        keep_cols = key_pre + ["f_stat", "pvalue", "n_obs", "r2"]
        comp = mpre[keep_cols].merge(qpre[keep_cols], on=key_pre, how="outer", suffixes=("_m", "_q"))
        comp.to_csv(outdir / "model1a_mq_pretrend_compare.tsv", sep="\t", index=False)

    key_static = ["design", "table_group", "reform", "target", "target_key", "spec_name", "horizon"]
    if not monthly["static_twfe"].empty or not quarterly["static_twfe"].empty:
        keep_cols = key_static + ["beta", "se", "pvalue", "n_obs", "r2"]
        comp = monthly["static_twfe"][keep_cols].merge(quarterly["static_twfe"][keep_cols], on=key_static, how="outer", suffixes=("_m", "_q"))
        comp.to_csv(outdir / "model1a_mq_static_compare.tsv", sep="\t", index=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-frequency Model 1A runner")
    parser.add_argument("--lp", default=None, help="Path to mixed-frequency LP_Panel.tsv")
    parser.add_argument("--out", default="Design/Output (new)/Model_1A_v6", help="Output directory")
    parser.add_argument("--run", choices=["monthly", "quarterly", "both"], default="both", help="Which frequency branch to run")
    args = parser.parse_args()

    lp_path = resolve_lp_path(args.lp)
    print(f"[Model_1A_v6] Using LP panel: {lp_path}")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_lp_panel(lp_path)
    results = {}

    if args.run in ("monthly", "both"):
        print("[Model_1A_v6] Building monthly analysis panel...")
        m_panel = build_monthly_panel(df)
        results["monthly"] = run_frequency(m_panel, "monthly", outdir)

    if args.run in ("quarterly", "both"):
        print("[Model_1A_v6] Building quarterly analysis panel...")
        q_panel = build_quarterly_panel(df)
        results["quarterly"] = run_frequency(q_panel, "quarterly", outdir)

    if args.run == "both":
        build_comparisons(results["monthly"], results["quarterly"], outdir)

    manifest = {
        "script": "Model_1A_v6_dual_frequency.py",
        "lp_input": str(lp_path),
        "run_mode": args.run,
        "monthly_bins": [b.__dict__ for b in MONTHLY_BINS],
        "quarterly_bins": [b.__dict__ for b in QUARTERLY_BINS],
        "monthly_windows": MONTHLY_WINDOWS,
        "quarterly_windows": QUARTERLY_WINDOWS,
        "notes": [
            "Monthly branch expands native quarterly terminal LP into within-quarter monthly step functions for comparison against the older monthly shell.",
            "Quarterly branch keeps post-reform terminal LP at native quarterly frequency.",
            "Both branches read the same mixed-frequency LP panel and therefore the same updated L/LP proxy branch.",
            "This file also fixes the quarter parsing bug by accepting quarter labels such as Q3.",
        ],
    }
    with open(outdir / "model1a_v6_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Model_1A_v6] Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
