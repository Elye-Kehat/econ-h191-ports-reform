
from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# =============================================================================
# Model_1B_v5_dynamic.py
#
# Haifa-only dynamic K/L extension for the thesis.
#
# Main idea:
#   - treat Model 1B as a descriptive dynamic extension rather than a clean
#     cross-port DiD
#   - estimate full month-by-month event-time paths for ln(K/L)
#   - summarize those paths into interpretable windows for the paper tables
#
# Main series:
#   - Haifa_Legacy_KL          (main text)
#   - Haifa_port_KL            (main text)
#   - Haifa_Bayport_KL         (supplementary appendix)
#
# Reform clocks:
#   - competition   : 2021-09 (Bayport entry)
#   - privatization : 2023-01 (Haifa privatization)
#
# Specifications:
#   1) baseline
#      log_KL ~ event-time bins
#
#   2) ctrl_trend
#      log_KL ~ event-time bins + linear trend + covid_shock + war_shock
#
# Inference:
#   - HAC / Newey-West standard errors
#
# Outputs:
#   Design/Output (new)/Model_1B_v5_dynamic/
#       model1b_dynamic_betas_all.tsv
#       model1b_window_betas_summary.tsv
#       model1b_pretrend_tests.tsv
#       model1b_sample_overview.tsv
#       model1b_plot_helper.tsv
#       model1b_table_cells_main.tsv
#       model1b_table_cells_bayport_appendix.tsv
#       model1b_manifest.json
# =============================================================================


# -----------------------------
# Configuration
# -----------------------------

MAIN_SERIES = ["Haifa_Legacy_KL", "Haifa_port_KL"]
SUPPLEMENTARY_SERIES = ["Haifa_Bayport_KL"]
SERIES_DISPLAY = {
    "Haifa_Legacy_KL": "Haifa-Legacy",
    "Haifa_port_KL": "Haifa aggregate",
    "Haifa_Bayport_KL": "Haifa-Bayport",
}

REFORMS = {
    "competition": {"year": 2021, "month": 9, "display": "Competition clock"},
    "privatization": {"year": 2023, "month": 1, "display": "Privatization clock"},
}

EXACT_MIN = -12
EXACT_MAX = 24
TAIL_LEFT = f"<= {EXACT_MIN - 1}"
TAIL_RIGHT = f">= {EXACT_MAX + 1}"
REFERENCE_BIN = "-1"

WINDOWS = [
    ("avg_pre", -12, -2, "Average pre"),
    ("post_1_6", 1, 6, "Average post, months 1-6"),
    ("post_7_12", 7, 12, "Average post, months 7-12"),
    ("post_13_24", 13, 24, "Average post, months 13-24"),
    ("post_full", 1, 24, "Average full post"),
]

REQUIRED_COLS = {
    "series_id", "year", "month", "K", "L", "KL", "log_KL",
}

SHOCK_WINDOWS = {
    "covid_shock": {"start": (2020, 1), "end": (2021, 12)},
    "war_shock": {"start": (2023, 10), "end": None},
}


# -----------------------------
# Helpers
# -----------------------------

def find_thesis_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2]


def month_serial(year: int, month: int) -> int:
    return int(year) * 12 + int(month)


def sanitize_filename(s: str) -> str:
    import re
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s


def stars_from_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p <= 0.01:
        return "***"
    if p <= 0.05:
        return "**"
    if p <= 0.10:
        return "*"
    return ""


def fmt_num(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def fmt_intlike(x: float) -> str:
    if not np.isfinite(x):
        return ""
    return str(int(round(float(x))))


def fmt_estimate(beta: float, se: float, pvalue: float) -> str:
    if not np.isfinite(beta):
        return ""
    stars = stars_from_p(pvalue)
    if np.isfinite(se):
        return f"{beta:.3f}{stars} ({se:.3f})"
    return f"{beta:.3f}{stars}"


def json_default(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def clear_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    patterns = [
        "model1b_*.tsv",
        "model1b_*.json",
    ]
    removed = 0
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok=True)
            removed += 1
    print(f"Cleared {removed} old Model 1B v5 output files from: {output_dir}")


def add_shock_controls(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    serial = out["year"].astype(int) * 12 + out["month"].astype(int)

    covid_start = month_serial(*SHOCK_WINDOWS["covid_shock"]["start"])
    covid_end = month_serial(*SHOCK_WINDOWS["covid_shock"]["end"])
    out["covid_shock"] = ((serial >= covid_start) & (serial <= covid_end)).astype(int)

    war_start = month_serial(*SHOCK_WINDOWS["war_shock"]["start"])
    out["war_shock"] = (serial >= war_start).astype(int)

    return out


def add_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype(int)
    out["month_serial"] = out["year"] * 12 + out["month"]
    out["date"] = pd.to_datetime(dict(year=out["year"], month=out["month"], day=1))
    out["t_linear"] = out["month_serial"] - int(out["month_serial"].min())
    return out


def load_kl_panel(path: Path) -> pd.DataFrame:
    print(f"Reading KL panel from: {path}")
    df = pd.read_csv(path, sep="\t")
    missing = REQUIRED_COLS.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")

    df = add_time_cols(df)
    df = add_shock_controls(df)

    expected = set(MAIN_SERIES + SUPPLEMENTARY_SERIES)
    actual = set(df["series_id"].dropna().astype(str).unique().tolist())
    if actual != expected:
        raise ValueError(
            "Unexpected KL series universe.\n"
            f"Expected: {sorted(expected)}\n"
            f"Actual:   {sorted(actual)}"
        )

    if (df["KL"] <= 0).any():
        bad = df.loc[df["KL"] <= 0, ["series_id", "year", "month", "KL"]].head(10)
        raise ValueError(f"KL must be positive. Example bad rows:\n{bad}")

    # make sure log_KL exists and is consistent enough
    if "log_KL" not in df.columns:
        df["log_KL"] = np.log(df["KL"])
    else:
        df["log_KL"] = pd.to_numeric(df["log_KL"], errors="coerce")
        miss = df["log_KL"].isna()
        if miss.any():
            df.loc[miss, "log_KL"] = np.log(df.loc[miss, "KL"])

    print(f"Loaded {len(df)} rows across {df['series_id'].nunique()} series.")
    return df


def event_bin_from_m(m: int) -> str:
    if m <= EXACT_MIN - 1:
        return TAIL_LEFT
    if m >= EXACT_MAX + 1:
        return TAIL_RIGHT
    return str(int(m))


def make_event_data(df_series: pd.DataFrame, reform: str) -> pd.DataFrame:
    info = REFORMS[reform]
    event_serial = month_serial(info["year"], info["month"])
    out = df_series.copy()
    out["event_time"] = out["month_serial"] - event_serial
    out["event_bin"] = out["event_time"].map(event_bin_from_m).astype(str)
    return out


def fit_hac_ols(formula: str, data: pd.DataFrame, hac_lags: int):
    model = smf.ols(formula=formula, data=data, missing="drop")
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})


def build_formula(spec_name: str) -> str:
    base = "log_KL ~ C(event_bin, Treatment(reference='-1'))"
    if spec_name == "baseline":
        return base
    if spec_name == "ctrl_trend":
        return base + " + t_linear + covid_shock + war_shock"
    raise ValueError(f"Unknown spec_name: {spec_name}")


def param_name_for_event_bin(event_bin: str) -> str:
    return f"C(event_bin, Treatment(reference='-1'))[T.{event_bin}]"


def extract_dynamic_rows(result, series_id: str, reform: str, spec_name: str) -> pd.DataFrame:
    rows = []
    params = result.params
    bse = result.bse
    pvals = result.pvalues

    # tail bins first
    for event_bin, event_kind, event_time in [
        (TAIL_LEFT, "tail_left", np.nan),
        (TAIL_RIGHT, "tail_right", np.nan),
    ]:
        name = param_name_for_event_bin(event_bin)
        rows.append({
            "series_id": series_id,
            "series_display": SERIES_DISPLAY[series_id],
            "is_main_series": int(series_id in MAIN_SERIES),
            "reform": reform,
            "reform_display": REFORMS[reform]["display"],
            "spec_name": spec_name,
            "event_bin": event_bin,
            "event_kind": event_kind,
            "event_time": event_time,
            "is_reference": 0,
            "beta": float(params.get(name, np.nan)),
            "se": float(bse.get(name, np.nan)),
            "pvalue": float(pvals.get(name, np.nan)),
            "n_obs": int(result.nobs),
            "r2": float(result.rsquared),
        })

    # exact monthly bins
    for m in range(EXACT_MIN, EXACT_MAX + 1):
        if m == -1:
            rows.append({
                "series_id": series_id,
                "series_display": SERIES_DISPLAY[series_id],
                "is_main_series": int(series_id in MAIN_SERIES),
                "reform": reform,
                "reform_display": REFORMS[reform]["display"],
                "spec_name": spec_name,
                "event_bin": str(m),
                "event_kind": "reference",
                "event_time": int(m),
                "is_reference": 1,
                "beta": 0.0,
                "se": 0.0,
                "pvalue": np.nan,
                "n_obs": int(result.nobs),
                "r2": float(result.rsquared),
            })
            continue

        name = param_name_for_event_bin(str(m))
        rows.append({
            "series_id": series_id,
            "series_display": SERIES_DISPLAY[series_id],
            "is_main_series": int(series_id in MAIN_SERIES),
            "reform": reform,
            "reform_display": REFORMS[reform]["display"],
            "spec_name": spec_name,
            "event_bin": str(m),
            "event_kind": "exact",
            "event_time": int(m),
            "is_reference": 0,
            "beta": float(params.get(name, np.nan)),
            "se": float(bse.get(name, np.nan)),
            "pvalue": float(pvals.get(name, np.nan)),
            "n_obs": int(result.nobs),
            "r2": float(result.rsquared),
        })

    return pd.DataFrame(rows)


def compute_window_averages(result, series_id: str, reform: str, spec_name: str) -> pd.DataFrame:
    params = result.params
    cov = result.cov_params()
    index_names = list(params.index)

    m_to_name = {}
    for m in range(EXACT_MIN, EXACT_MAX + 1):
        if m == -1:
            continue
        name = param_name_for_event_bin(str(m))
        if name in index_names:
            m_to_name[m] = name

    rows = []
    available_ms = sorted(m_to_name.keys())

    # detect actual data support
    reform_serial = month_serial(REFORMS[reform]["year"], REFORMS[reform]["month"])
    if reform == "competition":
        max_actual_post = month_serial(2024, 12) - reform_serial
    else:
        max_actual_post = month_serial(2024, 12) - reform_serial

    min_actual_pre = max(EXACT_MIN, -999)

    for window_key, a_req, b_req, window_label in WINDOWS:
        ms = [m for m in available_ms if a_req <= m <= b_req]
        if not ms:
            rows.append({
                "series_id": series_id,
                "series_display": SERIES_DISPLAY[series_id],
                "is_main_series": int(series_id in MAIN_SERIES),
                "reform": reform,
                "reform_display": REFORMS[reform]["display"],
                "spec_name": spec_name,
                "window_key": window_key,
                "window_label": window_label,
                "a_req": int(a_req),
                "b_req": int(b_req),
                "a_used": np.nan,
                "b_used": np.nan,
                "n_months_used": 0,
                "beta": np.nan,
                "se": np.nan,
                "pvalue": np.nan,
                "n_obs": int(result.nobs),
                "r2": float(result.rsquared),
            })
            continue

        w = pd.Series(0.0, index=index_names)
        for m in ms:
            w[m_to_name[m]] = 1.0 / len(ms)

        beta = float(np.dot(w.values, params.values))
        var = float(np.dot(w.values, np.dot(cov.values, w.values)))
        se = float(np.sqrt(var)) if var >= 0 else np.nan

        if np.isfinite(se) and se > 0:
            t_stat = beta / se
            cdf = 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0)))
            pvalue = 2.0 * (1.0 - cdf)
        else:
            pvalue = np.nan

        rows.append({
            "series_id": series_id,
            "series_display": SERIES_DISPLAY[series_id],
            "is_main_series": int(series_id in MAIN_SERIES),
            "reform": reform,
            "reform_display": REFORMS[reform]["display"],
            "spec_name": spec_name,
            "window_key": window_key,
            "window_label": window_label,
            "a_req": int(a_req),
            "b_req": int(b_req),
            "a_used": int(min(ms)),
            "b_used": int(max(ms)),
            "n_months_used": int(len(ms)),
            "beta": beta,
            "se": se,
            "pvalue": pvalue,
            "n_obs": int(result.nobs),
            "r2": float(result.rsquared),
        })

    return pd.DataFrame(rows)


def compute_pretrend_test(result, series_id: str, reform: str, spec_name: str) -> pd.DataFrame:
    params = result.params
    names = list(params.index)
    k = len(names)

    lead_indices = []
    lead_months = []
    for m in range(-12, -1):
        if m == -1:
            continue
        name = param_name_for_event_bin(str(m))
        if name in names:
            lead_indices.append(names.index(name))
            lead_months.append(m)

    if not lead_indices:
        return pd.DataFrame([{
            "series_id": series_id,
            "series_display": SERIES_DISPLAY[series_id],
            "is_main_series": int(series_id in MAIN_SERIES),
            "reform": reform,
            "reform_display": REFORMS[reform]["display"],
            "spec_name": spec_name,
            "n_leads_used": 0,
            "lead_min": np.nan,
            "lead_max": np.nan,
            "f_stat": np.nan,
            "pvalue": np.nan,
            "df_num": np.nan,
            "df_denom": np.nan,
            "n_obs": int(result.nobs),
            "r2": float(result.rsquared),
        }])

    R = np.zeros((len(lead_indices), k))
    for row_i, idx in enumerate(lead_indices):
        R[row_i, idx] = 1.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wt = result.wald_test(R, use_f=True, scalar=False)
        f_raw = getattr(wt, "fvalue", getattr(wt, "statistic", np.nan))
        f_val = float(np.asarray(f_raw).ravel()[0])
        p_raw = getattr(wt, "pvalue", np.nan)
        p_val = float(np.asarray(p_raw).ravel()[0])
        df_num = float(getattr(wt, "df_num", len(lead_indices)))
        df_denom = float(getattr(wt, "df_denom", result.df_resid))
    except Exception:
        f_val = np.nan
        p_val = np.nan
        df_num = float(len(lead_indices))
        df_denom = float(result.df_resid)

    return pd.DataFrame([{
        "series_id": series_id,
        "series_display": SERIES_DISPLAY[series_id],
        "is_main_series": int(series_id in MAIN_SERIES),
        "reform": reform,
        "reform_display": REFORMS[reform]["display"],
        "spec_name": spec_name,
        "n_leads_used": int(len(lead_indices)),
        "lead_min": int(min(lead_months)),
        "lead_max": int(max(lead_months)),
        "f_stat": f_val,
        "pvalue": p_val,
        "df_num": df_num,
        "df_denom": df_denom,
        "n_obs": int(result.nobs),
        "r2": float(result.rsquared),
    }])


def sample_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for series_id, g in df.groupby("series_id"):
        g = g.sort_values(["year", "month"])
        start_y = int(g["year"].iloc[0])
        start_m = int(g["month"].iloc[0])
        end_y = int(g["year"].iloc[-1])
        end_m = int(g["month"].iloc[-1])

        for reform, info in REFORMS.items():
            ev_serial = month_serial(info["year"], info["month"])
            et = g["month_serial"] - ev_serial
            rows.append({
                "series_id": series_id,
                "series_display": SERIES_DISPLAY[series_id],
                "is_main_series": int(series_id in MAIN_SERIES),
                "start_year": start_y,
                "start_month": start_m,
                "end_year": end_y,
                "end_month": end_m,
                "n_obs": int(len(g)),
                "reform": reform,
                "reform_display": info["display"],
                "event_time_min_observed": int(et.min()),
                "event_time_max_observed": int(et.max()),
                "supports_pre_full": int((et <= -2).sum() >= 11),
                "supports_1_6": int(((et >= 1) & (et <= 6)).sum() >= 6),
                "supports_7_12": int(((et >= 7) & (et <= 12)).sum() >= 6),
                "supports_13_24": int(((et >= 13) & (et <= 24)).sum() >= 12),
                "supports_1_24": int(((et >= 1) & (et <= 24)).sum() >= 24),
            })
    return pd.DataFrame(rows)


def build_table_cells(
    windows_df: pd.DataFrame,
    pre_df: pd.DataFrame,
    main_only: bool,
) -> pd.DataFrame:
    if main_only:
        series_keep = MAIN_SERIES
    else:
        series_keep = SUPPLEMENTARY_SERIES

    win = windows_df[windows_df["series_id"].isin(series_keep)].copy()
    pre = pre_df[pre_df["series_id"].isin(series_keep)].copy()

    if win.empty:
        return pd.DataFrame()

    spec_display = {
        "baseline": "Baseline",
        "ctrl_trend": "Controls+Trend",
    }

    reform_order = {"competition": 1, "privatization": 2}
    series_order = {
        "Haifa_Legacy_KL": 1,
        "Haifa_port_KL": 2,
        "Haifa_Bayport_KL": 1,
    }
    spec_order = {"baseline": 1, "ctrl_trend": 2}
    row_order = {
        "avg_pre": 1,
        "post_1_6": 2,
        "post_7_12": 3,
        "post_13_24": 4,
        "post_full": 5,
        "pretrend_p": 6,
        "n_obs": 7,
        "r2": 8,
    }

    rows = []

    # estimate rows
    for _, r in win.iterrows():
        col_key = f"{r['series_id']}__{r['spec_name']}"
        col_order = (series_order.get(r["series_id"], 99) - 1) * 2 + spec_order.get(r["spec_name"], 99)
        rows.append({
            "table_group": "main" if main_only else "bayport_appendix",
            "reform": r["reform"],
            "reform_order": reform_order.get(r["reform"], 99),
            "row_key": r["window_key"],
            "row_label": r["window_label"],
            "row_order": row_order.get(r["window_key"], 99),
            "row_type": "estimate",
            "series_id": r["series_id"],
            "series_display": r["series_display"],
            "spec_name": r["spec_name"],
            "spec_display": spec_display.get(r["spec_name"], r["spec_name"]),
            "column_key": col_key,
            "column_label": f"{r['series_display']} — {spec_display.get(r['spec_name'], r['spec_name'])}",
            "column_order": col_order,
            "beta": r["beta"],
            "se": r["se"],
            "pvalue": r["pvalue"],
            "n_obs": r["n_obs"],
            "r2": r["r2"],
            "a_used": r["a_used"],
            "b_used": r["b_used"],
            "n_months_used": r["n_months_used"],
            "value_display": fmt_estimate(r["beta"], r["se"], r["pvalue"]),
        })

    # summary rows from pretrend + full post windows
    full_post = win[win["window_key"] == "post_full"].copy()
    for _, r in pre.iterrows():
        col_key = f"{r['series_id']}__{r['spec_name']}"
        col_order = (series_order.get(r["series_id"], 99) - 1) * 2 + spec_order.get(r["spec_name"], 99)
        rows.append({
            "table_group": "main" if main_only else "bayport_appendix",
            "reform": r["reform"],
            "reform_order": reform_order.get(r["reform"], 99),
            "row_key": "pretrend_p",
            "row_label": "Pre-period joint p-value",
            "row_order": row_order["pretrend_p"],
            "row_type": "summary",
            "series_id": r["series_id"],
            "series_display": r["series_display"],
            "spec_name": r["spec_name"],
            "spec_display": spec_display.get(r["spec_name"], r["spec_name"]),
            "column_key": col_key,
            "column_label": f"{r['series_display']} — {spec_display.get(r['spec_name'], r['spec_name'])}",
            "column_order": col_order,
            "n_obs": r["n_obs"],
            "r2": r["r2"],
            "value_display": fmt_num(r["pvalue"], 3),
        })

    for _, r in full_post.iterrows():
        col_key = f"{r['series_id']}__{r['spec_name']}"
        col_order = (series_order.get(r["series_id"], 99) - 1) * 2 + spec_order.get(r["spec_name"], 99)
        rows.append({
            "table_group": "main" if main_only else "bayport_appendix",
            "reform": r["reform"],
            "reform_order": reform_order.get(r["reform"], 99),
            "row_key": "n_obs",
            "row_label": "Observations",
            "row_order": row_order["n_obs"],
            "row_type": "summary",
            "series_id": r["series_id"],
            "series_display": r["series_display"],
            "spec_name": r["spec_name"],
            "spec_display": spec_display.get(r["spec_name"], r["spec_name"]),
            "column_key": col_key,
            "column_label": f"{r['series_display']} — {spec_display.get(r['spec_name'], r['spec_name'])}",
            "column_order": col_order,
            "n_obs": r["n_obs"],
            "r2": r["r2"],
            "value_display": fmt_intlike(r["n_obs"]),
        })
        rows.append({
            "table_group": "main" if main_only else "bayport_appendix",
            "reform": r["reform"],
            "reform_order": reform_order.get(r["reform"], 99),
            "row_key": "r2",
            "row_label": "R^2",
            "row_order": row_order["r2"],
            "row_type": "summary",
            "series_id": r["series_id"],
            "series_display": r["series_display"],
            "spec_name": r["spec_name"],
            "spec_display": spec_display.get(r["spec_name"], r["spec_name"]),
            "column_key": col_key,
            "column_label": f"{r['series_display']} — {spec_display.get(r['spec_name'], r['spec_name'])}",
            "column_order": col_order,
            "n_obs": r["n_obs"],
            "r2": r["r2"],
            "value_display": fmt_num(r["r2"], 3),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["reform_order", "row_order", "column_order"]).reset_index(drop=True)
    return out


def estimate_one_series_reform(
    df_series: pd.DataFrame,
    series_id: str,
    reform: str,
    spec_name: str,
    hac_lags: int,
):
    d = make_event_data(df_series, reform)
    formula = build_formula(spec_name)
    result = fit_hac_ols(formula=formula, data=d, hac_lags=hac_lags)

    dyn = extract_dynamic_rows(result, series_id, reform, spec_name)
    win = compute_window_averages(result, series_id, reform, spec_name)
    pre = compute_pretrend_test(result, series_id, reform, spec_name)

    return result, dyn, win, pre


def main() -> None:
    parser = argparse.ArgumentParser(description="Haifa-only dynamic Model 1B extension.")
    parser.add_argument(
        "--kl",
        type=str,
        default="",
        help="Optional explicit path to KL_Panel_monthly.tsv. Defaults to Data/KL/KL_Panel_monthly.tsv relative to Thesis root.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Optional explicit output directory. Defaults to Design/Output (new)/Model_1B_v5_dynamic relative to Thesis root.",
    )
    parser.add_argument(
        "--hac-lags",
        type=int,
        default=6,
        help="HAC / Newey-West lag length. Default: 6",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear old files in the output directory before writing new ones.",
    )
    args = parser.parse_args()

    thesis_root = find_thesis_root()
    kl_path = Path(args.kl) if args.kl else thesis_root / "Data" / "KL" / "KL_Panel_monthly.tsv"
    output_dir = Path(args.outdir) if args.outdir else thesis_root / "Design" / "Output (new)" / "Model_1B_v5_dynamic"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_clear:
        clear_outputs(output_dir)

    df = load_kl_panel(kl_path)
    overview = sample_overview(df)

    all_dynamic = []
    all_windows = []
    all_pre = []

    specs = ["baseline", "ctrl_trend"]
    series_list = MAIN_SERIES + SUPPLEMENTARY_SERIES

    for series_id in series_list:
        df_series = df[df["series_id"] == series_id].copy().sort_values(["year", "month"])
        print(f"\n=== Series: {series_id} ({len(df_series)} rows) ===")
        for reform in REFORMS.keys():
            print(f"  Reform: {reform}")
            for spec_name in specs:
                print(f"    Spec: {spec_name}")
                _, dyn, win, pre = estimate_one_series_reform(
                    df_series=df_series,
                    series_id=series_id,
                    reform=reform,
                    spec_name=spec_name,
                    hac_lags=args.hac_lags,
                )
                all_dynamic.append(dyn)
                all_windows.append(win)
                all_pre.append(pre)

    dynamic_all = pd.concat(all_dynamic, ignore_index=True) if all_dynamic else pd.DataFrame()
    windows_all = pd.concat(all_windows, ignore_index=True) if all_windows else pd.DataFrame()
    pre_all = pd.concat(all_pre, ignore_index=True) if all_pre else pd.DataFrame()

    # Plot helper = exact monthly rows for all series / reforms / specs.
    plot_helper = dynamic_all[
        (dynamic_all["event_kind"] == "exact") &
        (dynamic_all["event_time"].notna())
    ].copy().sort_values(["series_id", "reform", "spec_name", "event_time"])

    table_main = build_table_cells(windows_all, pre_all, main_only=True)
    table_bayport = build_table_cells(windows_all, pre_all, main_only=False)

    dynamic_path = output_dir / "model1b_dynamic_betas_all.tsv"
    windows_path = output_dir / "model1b_window_betas_summary.tsv"
    pre_path = output_dir / "model1b_pretrend_tests.tsv"
    overview_path = output_dir / "model1b_sample_overview.tsv"
    plot_path = output_dir / "model1b_plot_helper.tsv"
    main_table_path = output_dir / "model1b_table_cells_main.tsv"
    bayport_table_path = output_dir / "model1b_table_cells_bayport_appendix.tsv"
    manifest_path = output_dir / "model1b_manifest.json"

    dynamic_all.to_csv(dynamic_path, sep="\t", index=False)
    windows_all.to_csv(windows_path, sep="\t", index=False)
    pre_all.to_csv(pre_path, sep="\t", index=False)
    overview.to_csv(overview_path, sep="\t", index=False)
    plot_helper.to_csv(plot_path, sep="\t", index=False)
    table_main.to_csv(main_table_path, sep="\t", index=False)
    table_bayport.to_csv(bayport_table_path, sep="\t", index=False)

    manifest = {
        "thesis_root": str(thesis_root),
        "kl_input": str(kl_path),
        "output_dir": str(output_dir),
        "hac_lags": int(args.hac_lags),
        "series_main": MAIN_SERIES,
        "series_supplementary": SUPPLEMENTARY_SERIES,
        "reforms": REFORMS,
        "exact_event_range": [EXACT_MIN, EXACT_MAX],
        "tail_bins": [TAIL_LEFT, TAIL_RIGHT],
        "reference_bin": REFERENCE_BIN,
        "windows": [
            {"window_key": k, "a_req": a, "b_req": b, "label": lab}
            for (k, a, b, lab) in WINDOWS
        ],
        "specs": [
            {"spec_name": "baseline", "formula": build_formula("baseline")},
            {"spec_name": "ctrl_trend", "formula": build_formula("ctrl_trend")},
        ],
        "outputs": {
            "dynamic_betas_all": str(dynamic_path),
            "window_betas_summary": str(windows_path),
            "pretrend_tests": str(pre_path),
            "sample_overview": str(overview_path),
            "plot_helper": str(plot_path),
            "table_cells_main": str(main_table_path),
            "table_cells_bayport_appendix": str(bayport_table_path),
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=json_default)

    print("\n=== Model_1B_v5_dynamic: done ===")
    print(f"Wrote dynamic betas  : {dynamic_path}")
    print(f"Wrote window summary : {windows_path}")
    print(f"Wrote pretrend tests : {pre_path}")
    print(f"Wrote sample overview: {overview_path}")
    print(f"Wrote plot helper    : {plot_path}")
    print(f"Wrote main table     : {main_table_path}")
    print(f"Wrote Bayport table  : {bayport_table_path}")
    print(f"Wrote manifest       : {manifest_path}")


if __name__ == "__main__":
    main()
