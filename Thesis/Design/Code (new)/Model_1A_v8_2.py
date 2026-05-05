#!/usr/bin/env python3
"""
Model_1A_v8.2.py

Purpose
-------
Quarterly-only Model 1A with the aggregate-port privatization pivot preserved,
the competition incumbent-splice mapping from v8.1, and a reporting-focused
upgrade for table production.

Why this version exists
-----------------------
Model_1A_v8.1 fixed the competition legacy mapping bug, but two reporting issues
remained:

1) the output tables only contained ordinary model R2 rather than a true
   within-R2 for the fixed-effects shells; and
2) the summary windows were too rigid across reforms, which meant the
   privatization branch did not surface the full supported seven post-treatment
   quarters in a clean table-ready way.

Main v8.2 upgrades
------------------
1) Preserve the v8.1 identification structure:
   - competition legacy targets use incumbent splices
   - aggregate-port privatization remains the headline privatization object

2) Compute and export both:
   - ordinary model R2
   - within-R2 based on the Frisch-Waugh-Lovell residualized regression
     after absorbing the nuisance FE/trend shell

3) Use reform-specific supported summary windows:
   - competition: post_y1, post_y2, post_y1_2, post_y3 when supported
   - privatization: post_y1, post_late, and full_post over the supported
     seven post-treatment quarters

4) Keep quarterly-only estimation and direct quarterly aggregate-port LP
   precedence unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


# -----------------------------------------------------------------------------
# Text / time helpers
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


def quarter_id(year: int, quarter: int) -> int:
    return int(year) * 4 + int(quarter) - 1


def quarter_label(year: int, quarter: int) -> str:
    return f"{int(year)}Q{int(quarter)}"


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
# Input resolution / loading
# -----------------------------------------------------------------------------

def resolve_lp_path(user_path: Optional[str]) -> Tuple[Path, str]:
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"Requested LP panel does not exist: {p}")
        return p, "explicit_cli"

    candidates = [
        (Path("Data/LP/raw_from_l_v6_tonsonly/LP_Panel.tsv"), "raw_from_l_v6_tonsonly"),
        (Path("Data/LP/LP_Panel.tsv"), "canonical_active"),
        (Path("Data/LP/raw_from_l_v5_tonsonly/LP_Panel.tsv"), "raw_from_l_v5_tonsonly_fallback"),
        (Path("Data/LP/raw_from_l_v3_tonsonly/LP_Panel.tsv"), "raw_from_l_v3_tonsonly_fallback"),
    ]
    for p, label in candidates:
        if p.exists():
            return p, label

    raise FileNotFoundError(
        "Could not resolve an LP_Panel.tsv path. "
        "Pass --lp explicitly, ideally Data/LP/raw_from_l_v6_tonsonly/LP_Panel.tsv."
    )


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

    df["terminal_canon"] = [
        canonical_terminal(p, t, s)
        for p, t, s in zip(df["port"], df["terminal"], df["series_id"])
    ]
    df["series_key"] = [
        canonical_series_key(p, l, t, s)
        for p, l, t, s in zip(df["port"], df["level"], df["terminal"], df["series_id"])
    ]
    df = df[df["series_key"].notna()].copy()
    df["log_LP"] = np.log(df["LP"].astype(float))
    return df


# -----------------------------------------------------------------------------
# Quarterly panel build
# -----------------------------------------------------------------------------

def quarter_from_month(month: int) -> int:
    return ((int(month) - 1) // 3) + 1


def build_quarterly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a single quarterly panel that combines:
      - monthly port LP, collapsed to quarter means of log(LP)
      - direct quarterly port LP rows from the LP builder
      - quarterly terminal LP, kept at native quarterly frequency

    Preference rule
    ---------------
    If a direct quarterly port row exists for a (series_key, quarter), it takes precedence over
    the monthly-collapsed port row for that same logical series and quarter. This avoids duplicate
    aggregate-port quarters once Haifa_port_Q / Ashdod_port_Q are present.
    """
    out_rows: List[pd.DataFrame] = []

    port_keys = {"haifa_port", "ashdod_port"}
    term_keys = {"haifa_legacy", "haifa_bayport", "ashdod_legacy", "ashdod_hct"}

    port_q = df[
        df["series_key"].isin(port_keys)
        & df["quarter"].notna()
        & df["month"].isna()
    ].copy()
    direct_port_keys = set()
    if not port_q.empty:
        port_q["quarter_num"] = port_q["quarter"].apply(parse_quarter_value)
        q = port_q[["series_key", "port", "year", "quarter_num", "log_LP"]].copy()
        q["terminal"] = ""
        q["n_obs"] = 1
        q["time_id"] = [quarter_id(y, qn) for y, qn in zip(q["year"], q["quarter_num"])]
        q["time_str"] = [quarter_label(y, qn) for y, qn in zip(q["year"], q["quarter_num"])]
        q["analysis_freq"] = "quarterly"
        direct_port_keys = set(zip(q["series_key"], q["time_id"]))
        out_rows.append(
            q[[
                "series_key", "port", "terminal", "year", "quarter_num",
                "time_id", "time_str", "log_LP", "n_obs", "analysis_freq"
            ]]
        )

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
        if direct_port_keys:
            q = q[~q.apply(lambda r: (r["series_key"], r["time_id"]) in direct_port_keys, axis=1)].copy()
        out_rows.append(
            q[[
                "series_key", "port", "terminal", "year", "quarter_num",
                "time_id", "time_str", "log_LP", "n_obs", "analysis_freq"
            ]]
        )

    term_q = df[df["series_key"].isin(term_keys) & df["quarter"].notna()].copy()
    if not term_q.empty:
        term_q["quarter_num"] = term_q["quarter"].apply(parse_quarter_value)
        q = term_q[["series_key", "port", "terminal_canon", "year", "quarter_num", "log_LP"]].copy()
        q = q.rename(columns={"terminal_canon": "terminal"})
        q["n_obs"] = 1
        q["time_id"] = [quarter_id(y, qn) for y, qn in zip(q["year"], q["quarter_num"])]
        q["time_str"] = [quarter_label(y, qn) for y, qn in zip(q["year"], q["quarter_num"])]
        q["analysis_freq"] = "quarterly"
        out_rows.append(
            q[[
                "series_key", "port", "terminal", "year", "quarter_num",
                "time_id", "time_str", "log_LP", "n_obs", "analysis_freq"
            ]]
        )

    if not out_rows:
        raise ValueError("No quarterly rows could be constructed from LP panel")

    panel = pd.concat(out_rows, ignore_index=True, sort=False)
    panel = panel.sort_values(["series_key", "time_id", "n_obs"]).drop_duplicates(["series_key", "time_id", "terminal"], keep="first")
    panel = panel.reset_index(drop=True)
    return panel


# -----------------------------------------------------------------------------
# Analysis paths / treatment objects
# -----------------------------------------------------------------------------

def add_analysis_paths(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    native_keep = panel[
        panel["series_key"].isin(["haifa_bayport", "haifa_legacy", "ashdod_legacy"])
    ].copy()
    native_keep["analysis_unit"] = native_keep["series_key"]
    native_keep["analysis_port"] = native_keep["port"]
    native_keep["source_series"] = native_keep["series_key"]
    rows.append(native_keep)

    haifa_comp_time = quarter_id(2021, 3)
    ashdod_comp_time = quarter_id(2022, 4)

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

    h_agg = panel[panel["series_key"] == "haifa_port"].copy()
    h_agg["analysis_unit"] = "haifa_port_agg"
    h_agg["analysis_port"] = "Haifa"
    h_agg["source_series"] = h_agg["series_key"]
    rows.append(h_agg)

    a_agg = panel[panel["series_key"] == "ashdod_port"].copy()
    a_agg["analysis_unit"] = "ashdod_port_agg"
    a_agg["analysis_port"] = "Ashdod"
    a_agg["source_series"] = a_agg["series_key"]
    rows.append(a_agg)

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


QUARTERLY_BINS: Tuple[EventBin, ...] = (
    EventBin("pre_8_5", -8, -5),
    EventBin("pre_4_2", -4, -2),
    EventBin("q0", 0, 0),
    EventBin("q1", 1, 1),
    EventBin("q2", 2, 2),
    EventBin("q3", 3, 3),
    EventBin("q4", 4, 4),
    EventBin("y2_q5_8", 5, 8),
    EventBin("y3_q9_12", 9, 12),
    EventBin("y4p_q13_40", 13, 40),
)

SUMMARY_WINDOWS_BY_REFORM: Dict[str, List[Tuple[str, Tuple[int, int]]]] = {
    "competition": [
        ("avg_pre", (-4, -2)),
        ("post_y1", (1, 4)),
        ("post_y2", (5, 8)),
        ("post_y1_2", (1, 8)),
        ("post_y3", (9, 12)),
    ],
    "privatization": [
        ("avg_pre", (-4, -2)),
        ("post_y1", (1, 4)),
        ("post_late", (5, 7)),
    ],
}


def window_specs_for_design(design: ReformDesign) -> List[Tuple[str, Tuple[int, int]]]:
    if design.reform in {"haifa_comp", "ashdod_comp"}:
        return SUMMARY_WINDOWS_BY_REFORM["competition"]
    if design.reform == "haifa_priv":
        return SUMMARY_WINDOWS_BY_REFORM["privatization"]
    raise KeyError(f"Unknown reform when resolving summary windows: {design.reform}")


def build_designs() -> List[ReformDesign]:
    haifa_comp = quarter_id(2021, 3)
    ashdod_comp = quarter_id(2022, 4)
    haifa_priv = quarter_id(2023, 1)
    haifa_nyt_end = quarter_id(2022, 3)

    return [
        ReformDesign(
            "competition", "haifa_comp", "Haifa-Legacy", "haifa_legacy",
            ("ashdod_incumbent",), haifa_comp, haifa_nyt_end
        ),
        ReformDesign(
            "competition", "haifa_comp", "Haifa-Aggregate", "haifa_port_agg",
            ("ashdod_port_agg",), haifa_comp, haifa_nyt_end
        ),
        ReformDesign(
            "competition", "ashdod_comp", "Ashdod-Legacy", "ashdod_legacy",
            ("haifa_incumbent",), ashdod_comp, None
        ),
        ReformDesign(
            "competition", "ashdod_comp", "Ashdod-Aggregate", "ashdod_port_agg",
            ("haifa_port_agg",), ashdod_comp, None
        ),
        ReformDesign(
            "privatization", "haifa_priv", "Haifa-Aggregate", "haifa_port_agg",
            ("ashdod_port_agg",), haifa_priv, None
        ),
        ReformDesign(
            "privatization_diag", "haifa_priv", "Haifa-Legacy", "haifa_legacy",
            ("haifa_bayport",), haifa_priv, None
        ),
        ReformDesign(
            "privatization_diag", "haifa_priv", "Haifa-Bayport", "haifa_bayport_placebo",
            ("haifa_legacy",), haifa_priv, None
        ),
    ]


def target_unit_for_design(d: ReformDesign) -> str:
    """
    Reform-specific target mapping.

    Competition legacy targets should use the incumbent splice, not the raw post-reform legacy terminal.
    Privatization diagnostics should keep the raw terminal objects.
    """
    if d.reform == "haifa_comp" and d.target == "Haifa-Legacy":
        return "haifa_incumbent"
    if d.reform == "ashdod_comp" and d.target == "Ashdod-Legacy":
        return "ashdod_incumbent"
    if d.target == "Haifa-Bayport":
        return "haifa_bayport"
    if d.target == "Haifa-Aggregate":
        return "haifa_port_agg"
    if d.target == "Ashdod-Aggregate":
        return "ashdod_port_agg"
    if d.target == "Haifa-Legacy":
        return "haifa_legacy"
    if d.target == "Ashdod-Legacy":
        return "ashdod_legacy"
    raise KeyError(f"Unknown target mapping requested for reform={d.reform}, target={d.target}")


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
        out[col] = (
            (out["treated_unit"] == 1)
            & (out["event_time"] >= b.a)
            & (out["event_time"] <= b.b)
        ).astype(int)
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
    return res, y, X


def _residualize_against_nuisance(y: pd.DataFrame, X: pd.DataFrame, nuisance_cols: List[str], target_cols: List[str]):
    y_arr = np.asarray(y).reshape(-1, 1)
    target_cols = [c for c in target_cols if c in X.columns]
    X_target = X[target_cols].to_numpy(dtype=float) if target_cols else np.empty((len(X), 0))

    if nuisance_cols:
        Z = X[nuisance_cols].to_numpy(dtype=float)
        beta_y, *_ = np.linalg.lstsq(Z, y_arr, rcond=None)
        y_tilde = y_arr - Z @ beta_y
        if X_target.shape[1] > 0:
            beta_x, *_ = np.linalg.lstsq(Z, X_target, rcond=None)
            X_tilde = X_target - Z @ beta_x
        else:
            X_tilde = X_target
    else:
        y_tilde = y_arr.copy()
        X_tilde = X_target.copy()

    return y_tilde.reshape(-1), X_tilde, target_cols


def compute_within_r2(res, y: pd.DataFrame, X: pd.DataFrame, target_cols: List[str]) -> float:
    nuisance_cols = [c for c in X.columns if c not in target_cols]
    y_tilde, X_tilde, kept_targets = _residualize_against_nuisance(y, X, nuisance_cols, target_cols)
    if X_tilde.shape[1] == 0:
        return np.nan

    kept_targets = [c for c in kept_targets if c in res.params.index]
    if not kept_targets:
        return np.nan

    col_idx = [target_cols.index(c) for c in kept_targets]
    X_use = X_tilde[:, col_idx]
    beta = res.params[kept_targets].to_numpy(dtype=float)
    fitted = X_use @ beta
    resid = y_tilde - fitted
    tss = float(np.sum((y_tilde - y_tilde.mean()) ** 2))
    if not np.isfinite(tss) or tss <= 1e-12:
        return np.nan
    rss = float(np.sum(resid ** 2))
    return float(1.0 - rss / tss)


def contiguous_supported_post_max(treated_event_times: Sequence[int]) -> Optional[int]:
    post_set = {int(v) for v in treated_event_times if pd.notna(v) and int(v) >= 1}
    if not post_set:
        return None
    m = 1
    while m in post_set:
        m += 1
    return m - 1


def is_window_supported(window: Tuple[int, int], treated_event_times: Sequence[int]) -> bool:
    a, b = window
    treated_set = {int(v) for v in treated_event_times if pd.notna(v)}
    return all(q in treated_set for q in range(int(a), int(b) + 1))


def supported_window_specs(specs: Sequence[Tuple[str, Tuple[int, int]]], treated_event_times: Sequence[int]) -> List[Tuple[str, Tuple[int, int]]]:
    out: List[Tuple[str, Tuple[int, int]]] = []
    for name, ab in specs:
        if is_window_supported(ab, treated_event_times):
            out.append((name, ab))
    return out


def weighted_average_from_bins(
    params: pd.Series,
    cov: pd.DataFrame,
    window: Tuple[int, int],
    bins: Sequence[EventBin],
) -> Tuple[float, float, List[str]]:
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

    cols = list(weights.keys())
    total_w = float(sum(weights.values()))
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

def subset_for_design(
    panel: pd.DataFrame,
    design: ReformDesign,
    design_type: str,
    max_pre: int,
    max_post: int,
) -> pd.DataFrame:
    target_unit = target_unit_for_design(design)
    keep_units = {target_unit, *design.control_units}
    df = panel[panel["analysis_unit"].isin(keep_units)].copy()

    if design.reform == "haifa_priv" and design.target in {"Haifa-Legacy", "Haifa-Bayport"}:
        keep_units = {"haifa_legacy", "haifa_bayport"}
        df = panel[panel["analysis_unit"].isin(keep_units)].copy()

    df["event_time"] = df["time_id"] - design.reform_time
    df = df[(df["event_time"] >= -max_pre) & (df["event_time"] <= max_post)].copy()

    if design_type == "NYT" and design.nyt_end_time is not None:
        df = df[df["time_id"] <= design.nyt_end_time].copy()

    if design.reform == "haifa_priv" and design.target in {"Haifa-Legacy", "Haifa-Bayport"}:
        start = quarter_id(2021, 3)
        df = df[df["time_id"] >= start].copy()

    df = df.sort_values(["analysis_unit", "time_id"]).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Estimation
# -----------------------------------------------------------------------------

def run_dynamic_regression(
    df: pd.DataFrame,
    design: ReformDesign,
    design_type: str,
    spec_name: str,
    bins: Sequence[EventBin],
    windows: Sequence[Tuple[str, Tuple[int, int]]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_unit = target_unit_for_design(design)
    work = build_event_columns(df, target_unit, bins)
    work = add_trend_columns(work)

    dynamic_cols = [event_bin_name(b.label) for b in bins]
    formula = build_formula(work, dynamic_cols, spec_name)
    res, y, X = fit_ols(work, formula)
    within_r2 = compute_within_r2(res, y, X, dynamic_cols)

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
            "bin_label": eb.label,
            "a": eb.a,
            "b": eb.b,
            "beta": float(params[col]),
            "se": float(res.bse[col]),
            "t": float(res.tvalues[col]),
            "pvalue": float(res.pvalues[col]),
            "n_event_obs": treated_obs_in_bin,
            "n_obs": int(res.nobs),
            "r2": float(res.rsquared),
            "within_r2": within_r2,
            "se_type": "HC1",
        })

    treated_event_times = work.loc[work["treated_unit"] == 1, "event_time"].tolist()
    window_specs = supported_window_specs(windows, treated_event_times)

    win_rows = []
    for window_name, ab in window_specs:
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
            "within_r2": within_r2,
            "used_bins": ",".join(used_cols),
        })

    max_supported_post = contiguous_supported_post_max(treated_event_times)
    if max_supported_post is not None and max_supported_post >= 1:
        beta, se, used_cols = weighted_average_from_bins(params, cov, (1, max_supported_post), bins)
        win_rows.append({
            "design": design_type,
            "table_group": design.table_group,
            "reform": design.reform,
            "target": design.target,
            "target_key": design.target_key,
            "spec_name": spec_name,
            "window": "full_post",
            "a": 1,
            "b": max_supported_post,
            "beta": beta,
            "se": se,
            "n_obs": int(res.nobs),
            "r2": float(res.rsquared),
            "within_r2": within_r2,
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
        "within_r2": within_r2,
        "test_type": "chi2_wald_HC1",
    }

    return pd.DataFrame(dyn_rows), pd.DataFrame(win_rows), pd.DataFrame([pre_row])


def run_static_regression(
    df: pd.DataFrame,
    design: ReformDesign,
    spec_name: str,
    windows: Sequence[Tuple[str, Tuple[int, int]]],
) -> pd.DataFrame:
    target_unit = target_unit_for_design(design)
    base = add_trend_columns(df.copy())
    base["treated_unit"] = (base["analysis_unit"] == target_unit).astype(int)

    treated_event_times = base.loc[base["treated_unit"] == 1, "event_time"].tolist()
    max_supported_post = contiguous_supported_post_max(treated_event_times)

    horizons: List[Tuple[str, Tuple[int, int]]] = []
    for name, ab in windows:
        if name == "avg_pre":
            continue
        if is_window_supported(ab, treated_event_times):
            horizons.append((name, ab))
    if max_supported_post is not None and max_supported_post >= 1:
        horizons = [("full_post", (1, max_supported_post))] + horizons

    rows = []
    for horizon_name, (a, b) in horizons:
        work = base.copy()
        work = work[(work["event_time"] >= min(-8, int(base["event_time"].min()))) & (work["event_time"] <= b)].copy()
        work["treated_post"] = (
            (work["treated_unit"] == 1)
            & (work["event_time"] >= a)
            & (work["event_time"] <= b)
        ).astype(int)

        rhs = ["C(analysis_unit)"]
        if spec_name == "baseline":
            rhs.append("C(time_id)")
        else:
            rhs.extend([c for c in work.columns if c.startswith("trend_")])
        rhs.append("treated_post")

        formula = "log_LP ~ " + " + ".join(rhs)
        res, y, X = fit_ols(work, formula)

        if "treated_post" not in res.params.index:
            continue

        within_r2 = compute_within_r2(res, y, X, ["treated_post"])

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
            "within_r2": within_r2,
            "n_treated": int((work["treated_unit"] == 1).sum()),
            "n_control": int((work["treated_unit"] == 0).sum()),
            "n_post_treated": int(((work["treated_unit"] == 1) & (work["event_time"] >= a) & (work["event_time"] <= b)).sum()),
            "se_type": "HC1",
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def build_sample_overview(panel: pd.DataFrame) -> pd.DataFrame:
    grp = (
        panel.groupby(["analysis_unit", "analysis_port"], as_index=False)
        .agg(
            first_time_id=("time_id", "min"),
            last_time_id=("time_id", "max"),
            n_obs=("time_id", "size"),
            n_unique_periods=("time_id", "nunique"),
        )
        .sort_values(["analysis_unit"])
        .reset_index(drop=True)
    )
    return grp


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------

def run_quarterly_only(lp_path: Path, outdir: Path) -> Dict[str, pd.DataFrame]:
    df = load_lp_panel(lp_path)
    q_panel_native = build_quarterly_panel(df)
    q_panel = add_analysis_paths(q_panel_native)

    q_panel = q_panel[
        q_panel["analysis_unit"].isin(
            [
                "haifa_incumbent",
                "ashdod_incumbent",
                "haifa_legacy",
                "haifa_bayport",
                "haifa_port_agg",
                "ashdod_port_agg",
            ]
        )
    ].copy()

    q_panel.to_csv(outdir / "model1a_q_analysis_panel.tsv", sep="\t", index=False)
    build_sample_overview(q_panel).to_csv(outdir / "model1a_q_sample_overview.tsv", sep="\t", index=False)

    bins = QUARTERLY_BINS
    max_pre, max_post = 8, 40

    dynamic_nyt = []
    dynamic_twfe = []
    windows_nyt = []
    windows_twfe = []
    pretrend_nyt = []
    pretrend_twfe = []
    static_twfe = []

    for d in build_designs():
        windows = window_specs_for_design(d)

        if d.reform != "ashdod_comp":
            for spec_name in ("baseline", "porttr"):
                sample = subset_for_design(q_panel, d, "NYT", max_pre, max_post)
                if sample.empty:
                    continue
                dyn, win, pre = run_dynamic_regression(sample, d, "NYT", spec_name, bins, windows)
                dynamic_nyt.append(dyn)
                windows_nyt.append(win)
                pretrend_nyt.append(pre)

        for spec_name in ("baseline", "porttr"):
            sample = subset_for_design(q_panel, d, "TWFE", max_pre, max_post)
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

    out["dynamic_nyt"].to_csv(outdir / "model1a_q_dynamic_betas_nyt.tsv", sep="\t", index=False)
    out["dynamic_twfe"].to_csv(outdir / "model1a_q_dynamic_betas_twfe.tsv", sep="\t", index=False)
    out["windows_nyt"].to_csv(outdir / "model1a_q_window_betas_nyt.tsv", sep="\t", index=False)
    out["windows_twfe"].to_csv(outdir / "model1a_q_window_betas_twfe.tsv", sep="\t", index=False)
    out["pretrend_nyt"].to_csv(outdir / "model1a_q_pretrend_tests_nyt.tsv", sep="\t", index=False)
    out["pretrend_twfe"].to_csv(outdir / "model1a_q_pretrend_tests_twfe.tsv", sep="\t", index=False)
    out["static_twfe"].to_csv(outdir / "model1a_q_static_betas_twfe.tsv", sep="\t", index=False)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Quarterly-only Model 1A with aggregate-port privatization pivot")
    parser.add_argument("--lp", default=None, help="Path to mixed-frequency LP_Panel.tsv")
    parser.add_argument(
        "--out",
        default="Design/Output (new)/Model_1A_v8_2",
        help="Output directory",
    )
    args = parser.parse_args()

    lp_path, lp_resolution = resolve_lp_path(args.lp)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[Model_1A_v8.2] Using LP panel: {lp_path}")
    print(f"[Model_1A_v8.2] LP resolution mode: {lp_resolution}")
    print("[Model_1A_v8.2] Quarterly-only run: no monthly shell will be estimated.")

    run_quarterly_only(lp_path, outdir)

    manifest = {
        "script": "Model_1A_v8_2.py",
        "lp_input": str(lp_path),
        "lp_resolution": lp_resolution,
        "run_mode": "quarterly_only",
        "design_notes": [
            "Quarterly only: avoids pseudo-replication from repeating quarterly terminal LP across months.",
            "Reads direct quarterly aggregate-port LP rows when available and gives them precedence over monthly-collapsed port quarters.",
            "Creates continuous aggregate analysis units haifa_port_agg and ashdod_port_agg.",
            "Promotes Haifa aggregate-port privatization to the main privatization object.",
            "Fixes the competition legacy mapping so Haifa-Legacy and Ashdod-Legacy use incumbent splices under competition.",
            "Exports both ordinary model R2 and true within-R2 based on the residualized treatment regression.",
            "Uses reform-specific supported summary windows, including post_y1 / post_late / full_post for privatization.",
            "Keeps Legacy-versus-Bayport privatization objects only as diagnostics under table_group=privatization_diag.",
        ],
        "quarterly_bins": [b.__dict__ for b in QUARTERLY_BINS],
        "summary_windows_by_reform": SUMMARY_WINDOWS_BY_REFORM,
        "outputs_written": {
            "analysis_panel": "model1a_q_analysis_panel.tsv",
            "sample_overview": "model1a_q_sample_overview.tsv",
            "dynamic_nyt": "model1a_q_dynamic_betas_nyt.tsv",
            "dynamic_twfe": "model1a_q_dynamic_betas_twfe.tsv",
            "dynamic_plot_helper": "model1a_q_dynamic_plot_helper.tsv",
            "windows_nyt": "model1a_q_window_betas_nyt.tsv",
            "windows_twfe": "model1a_q_window_betas_twfe.tsv",
            "pretrend_nyt": "model1a_q_pretrend_tests_nyt.tsv",
            "pretrend_twfe": "model1a_q_pretrend_tests_twfe.tsv",
            "static_twfe": "model1a_q_static_betas_twfe.tsv",
        },
    }

    with open(outdir / "model1a_v8_1_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Model_1A_v8.2] Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
