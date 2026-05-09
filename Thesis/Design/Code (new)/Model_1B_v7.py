
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# =============================================================================
# Model_1B_v7.py
#
# Haifa-only K/L extension, v7.
#
# Main changes relative to v6:
#   1) Keep the exact-month event study only for plotting / descriptive dynamics.
#      We no longer use exact-month windows or exact pretrend tests for inference.
#
#   2) Keep the binned event-study as the main inferential family.
#
#   3) Upgrade overlap controls from simple post dummies to coarse dynamic overlap
#      bins for the other reform clock.
#
#   4) Suppress unsupported summary windows and pretrend tests (especially relevant
#      for Bayport under the competition clock).
#
#   5) Continue to estimate two specifications:
#        baseline    = event terms + dynamic overlap control
#        ctrl_trend  = event terms + dynamic overlap control + t_linear
#                      + covid_shock + war_shock
#
# Output folder:
#   Design/Output (new)/Model_1B_v7/
# =============================================================================


# -----------------------------
# Configuration
# -----------------------------

MAIN_SERIES = ["Haifa_Legacy_KL", "Haifa_port_KL"]
SUPPLEMENTARY_SERIES = ["Haifa_Bayport_KL"]
ALL_SERIES = MAIN_SERIES + SUPPLEMENTARY_SERIES

SERIES_DISPLAY = {
    "Haifa_Legacy_KL": "Haifa-Legacy",
    "Haifa_port_KL": "Aggregate port",
    "Haifa_Bayport_KL": "Haifa-Bayport",
}

REFORMS: Dict[str, Dict[str, object]] = {
    "competition": {
        "year": 2021,
        "month": 9,
        "display": "Competition clock",
        "support_min": -12,
        "support_max": 24,
        "windows": [
            ("avg_pre", -12, -2, "Average pre, months -12 to -2"),
            ("post_1_6", 1, 6, "Average post, months 1-6"),
            ("post_7_12", 7, 12, "Average post, months 7-12"),
            ("post_13_tail", 13, 24, "Average post, months 13-24"),
            ("post_full", 1, 24, "Average full post, months 1-24"),
        ],
        "bin_defs": [
            ("pre_12_7", -12, -7, "Lead bin [-12,-7]"),
            ("pre_6_2", -6, -2, "Lead bin [-6,-2]"),
            ("ref_m1", -1, -1, "Reference month -1"),
            ("m0", 0, 0, "Month 0"),
            ("post_1_6", 1, 6, "Post bin [1,6]"),
            ("post_7_12", 7, 12, "Post bin [7,12]"),
            ("post_13_tail", 13, 24, "Post bin [13,24]"),
        ],
        # Competition regressions control for the privatization clock dynamically.
        # Within the competition support window, privatization ET ranges from about
        # -28 to +8. We absorb this coarsely.
        "overlap_bins": [
            ("pre_priv", None, -1, "Privatization pre-period"),
            ("priv_0_6", 0, 6, "Privatization overlap [0,6]"),
            ("priv_7_plus", 7, None, "Privatization overlap [7,+]"),
        ],
        "overlap_reference": "pre_priv",
    },
    "privatization": {
        "year": 2023,
        "month": 1,
        "display": "Privatization clock",
        "support_min": -12,
        "support_max": 23,
        "windows": [
            ("avg_pre", -12, -2, "Average pre, months -12 to -2"),
            ("post_1_6", 1, 6, "Average post, months 1-6"),
            ("post_7_12", 7, 12, "Average post, months 7-12"),
            ("post_13_tail", 13, 23, "Average post, months 13-23"),
            ("post_full", 1, 23, "Average full post, months 1-23"),
        ],
        "bin_defs": [
            ("pre_12_7", -12, -7, "Lead bin [-12,-7]"),
            ("pre_6_2", -6, -2, "Lead bin [-6,-2]"),
            ("ref_m1", -1, -1, "Reference month -1"),
            ("m0", 0, 0, "Month 0"),
            ("post_1_6", 1, 6, "Post bin [1,6]"),
            ("post_7_12", 7, 12, "Post bin [7,12]"),
            ("post_13_tail", 13, 23, "Post bin [13,23]"),
        ],
        # Privatization regressions control for the already-ongoing competition
        # regime dynamically. In the privatization support window, competition ET
        # ranges from about +4 to +39.
        "overlap_bins": [
            ("comp_4_12", 4, 12, "Competition overlap [4,12]"),
            ("comp_13_24", 13, 24, "Competition overlap [13,24]"),
            ("comp_25_plus", 25, None, "Competition overlap [25,+]"),
        ],
        "overlap_reference": "comp_4_12",
    },
}

SPEC_LABELS = {
    "baseline": "Baseline",
    "ctrl_trend": "Controls+Trend",
}

REQUIRED_COLS = {"series_id", "year", "month", "K", "L", "KL", "log_KL"}

SHOCK_WINDOWS = {
    "covid_shock": {"start": (2020, 1), "end": (2021, 12)},
    "war_shock": {"start": (2023, 10), "end": None},
}


# -----------------------------
# Generic helpers
# -----------------------------


def find_thesis_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2]


def month_serial(year: int, month: int) -> int:
    return int(year) * 12 + int(month)


def fmt_num(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def fmt_intlike(x: float) -> str:
    if not np.isfinite(x):
        return ""
    return str(int(round(float(x))))


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
    removed = 0
    for pattern in ["model1b_*.tsv", "model1b_*.json"]:
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok=True)
            removed += 1
    print(f"Cleared {removed} old Model 1B v7 output files from: {output_dir}")


# -----------------------------
# Data preparation
# -----------------------------


def add_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype(int)
    out["month_serial"] = out["year"] * 12 + out["month"]
    out["date"] = pd.to_datetime(dict(year=out["year"], month=out["month"], day=1))
    out["t_linear"] = out["month_serial"] - int(out["month_serial"].min())
    return out


def add_shock_controls(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    serial = out["month_serial"]

    covid_start = month_serial(*SHOCK_WINDOWS["covid_shock"]["start"])
    covid_end = month_serial(*SHOCK_WINDOWS["covid_shock"]["end"])
    out["covid_shock"] = ((serial >= covid_start) & (serial <= covid_end)).astype(int)

    war_start = month_serial(*SHOCK_WINDOWS["war_shock"]["start"])
    out["war_shock"] = (serial >= war_start).astype(int)

    comp_start = month_serial(REFORMS["competition"]["year"], REFORMS["competition"]["month"])
    priv_start = month_serial(REFORMS["privatization"]["year"], REFORMS["privatization"]["month"])
    out["post_comp"] = (serial >= comp_start).astype(int)
    out["post_priv"] = (serial >= priv_start).astype(int)
    return out


def load_kl_panel(path: Path) -> pd.DataFrame:
    print(f"Reading KL panel from: {path}")
    df = pd.read_csv(path, sep="\t")
    missing = REQUIRED_COLS.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")

    df = add_time_cols(df)
    df = add_shock_controls(df)

    expected = set(ALL_SERIES)
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

    df["log_KL"] = pd.to_numeric(df["log_KL"], errors="coerce")
    miss = df["log_KL"].isna()
    if miss.any():
        df.loc[miss, "log_KL"] = np.log(df.loc[miss, "KL"])

    print(f"Loaded {len(df)} rows across {df['series_id'].nunique()} series.")
    return df


# -----------------------------
# Event-time construction
# -----------------------------


def prepare_reform_sample(df_series: pd.DataFrame, reform: str) -> pd.DataFrame:
    info = REFORMS[reform]
    event_serial = month_serial(int(info["year"]), int(info["month"]))
    support_min = int(info["support_min"])
    support_max = int(info["support_max"])

    out = df_series.copy()
    out["event_time"] = out["month_serial"] - event_serial
    out = out.loc[(out["event_time"] >= support_min) & (out["event_time"] <= support_max)].copy()
    out = add_overlap_tokens(out, reform)
    return out.sort_values(["year", "month"]).reset_index(drop=True)


def exact_token(m: int) -> str:
    return str(int(m))


def assign_binned_token(m: int, reform: str) -> Optional[str]:
    for bin_key, a, b, _label in REFORMS[reform]["bin_defs"]:
        if a <= m <= b:
            if bin_key == "ref_m1":
                return "-1"
            return str(bin_key)
    return None


def assign_overlap_token(other_event_time: int, reform: str) -> Optional[str]:
    for key, a, b, _label in REFORMS[reform]["overlap_bins"]:
        ok_lo = True if a is None else other_event_time >= a
        ok_hi = True if b is None else other_event_time <= b
        if ok_lo and ok_hi:
            return str(key)
    return None


def add_overlap_tokens(df: pd.DataFrame, reform: str) -> pd.DataFrame:
    out = df.copy()
    comp_serial = month_serial(int(REFORMS["competition"]["year"]), int(REFORMS["competition"]["month"]))
    priv_serial = month_serial(int(REFORMS["privatization"]["year"]), int(REFORMS["privatization"]["month"]))

    if reform == "competition":
        other_et = out["month_serial"] - priv_serial
    elif reform == "privatization":
        other_et = out["month_serial"] - comp_serial
    else:
        raise ValueError(f"Unknown reform: {reform}")

    out["overlap_event_time"] = other_et.astype(int)
    out["overlap_token"] = out["overlap_event_time"].astype(int).map(lambda m: assign_overlap_token(m, reform))
    return out


# -----------------------------
# Estimation helpers
# -----------------------------


def fit_hac_ols(formula: str, data: pd.DataFrame, hac_lags: int):
    model = smf.ols(formula=formula, data=data, missing="drop")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})
    return result


def build_formula(family: str, reform: str, spec_name: str) -> str:
    overlap_ref = REFORMS[reform]["overlap_reference"]
    overlap_part = f" + C(overlap_token, Treatment(reference='{overlap_ref}'))"

    if family == "exact":
        base = "log_KL ~ C(exact_token, Treatment(reference='-1'))"
    elif family == "binned":
        base = "log_KL ~ C(binned_token, Treatment(reference='-1'))"
    else:
        raise ValueError(f"Unknown family: {family}")

    base += overlap_part
    if spec_name == "baseline":
        return base
    if spec_name == "ctrl_trend":
        return base + " + t_linear + covid_shock + war_shock"
    raise ValueError(f"Unknown spec_name: {spec_name}")


def safe_wald_test(result, param_names: List[str]) -> Tuple[float, float, float, float]:
    names = list(result.params.index)
    chosen = [p for p in param_names if p in names]
    if not chosen:
        return np.nan, np.nan, np.nan, np.nan

    R = np.zeros((len(chosen), len(names)))
    for i, pname in enumerate(chosen):
        R[i, names.index(pname)] = 1.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wt = result.wald_test(R, use_f=True, scalar=False)
        f_raw = getattr(wt, "fvalue", getattr(wt, "statistic", np.nan))
        p_raw = getattr(wt, "pvalue", np.nan)
        df_num = float(getattr(wt, "df_num", len(chosen)))
        df_denom = float(getattr(wt, "df_denom", result.df_resid))
        f_val = float(np.asarray(f_raw).ravel()[0])
        p_val = float(np.asarray(p_raw).ravel()[0])
        return f_val, p_val, df_num, df_denom
    except Exception:
        return np.nan, np.nan, float(len(chosen)), float(result.df_resid)


def linear_combo_from_params(result, param_weights: Dict[str, float]) -> Tuple[float, float, float]:
    names = list(result.params.index)
    if not param_weights:
        return np.nan, np.nan, np.nan

    w = pd.Series(0.0, index=names)
    for pname, weight in param_weights.items():
        if pname in w.index:
            w[pname] = weight

    if float(np.abs(w).sum()) == 0.0:
        return np.nan, np.nan, np.nan

    beta = float(np.dot(w.values, result.params.reindex(names).values))
    cov = result.cov_params().reindex(index=names, columns=names).fillna(0.0)
    var = float(np.dot(w.values, np.dot(cov.values, w.values)))
    se = float(np.sqrt(var)) if var >= 0 else np.nan

    if np.isfinite(se) and se > 0:
        t_stat = beta / se
        cdf = 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0)))
        pvalue = 2.0 * (1.0 - cdf)
    else:
        pvalue = np.nan
    return beta, se, pvalue


# -----------------------------
# Exact family outputs
# -----------------------------


def exact_param_name(token: str) -> str:
    return f"C(exact_token, Treatment(reference='-1'))[T.{token}]"


def extract_exact_rows(result, sample: pd.DataFrame, series_id: str, reform: str, spec_name: str) -> pd.DataFrame:
    rows = []
    support_min = int(REFORMS[reform]["support_min"])
    support_max = int(REFORMS[reform]["support_max"])
    observed = set(sample["event_time"].astype(int).tolist())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bse = result.bse
        pvalues = result.pvalues

    for m in range(support_min, support_max + 1):
        token = exact_token(m)
        if m == -1:
            beta, se, pvalue, is_ref = 0.0, 0.0, np.nan, 1
        else:
            pname = exact_param_name(token)
            beta = float(result.params.get(pname, np.nan))
            se = float(bse.get(pname, np.nan))
            pvalue = float(pvalues.get(pname, np.nan))
            is_ref = 0

        rows.append({
            "family": "exact",
            "series_id": series_id,
            "series_display": SERIES_DISPLAY[series_id],
            "is_main_series": int(series_id in MAIN_SERIES),
            "reform": reform,
            "reform_display": REFORMS[reform]["display"],
            "spec_name": spec_name,
            "event_time": int(m),
            "event_token": token,
            "is_reference": is_ref,
            "observed_in_sample": int(m in observed),
            "beta": beta,
            "se": se,
            "pvalue": pvalue,
            "n_obs": int(result.nobs),
            "r2": float(result.rsquared),
        })

    return pd.DataFrame(rows)


# -----------------------------
# Binned family outputs
# -----------------------------


def binned_param_name(token: str) -> str:
    return f"C(binned_token, Treatment(reference='-1'))[T.{token}]"


def extract_binned_rows(result, sample: pd.DataFrame, series_id: str, reform: str, spec_name: str) -> pd.DataFrame:
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bse = result.bse
        pvalues = result.pvalues
    for bin_key, a, b, label in REFORMS[reform]["bin_defs"]:
        token = "-1" if bin_key == "ref_m1" else str(bin_key)
        bin_sample = sample.loc[sample["binned_token"] == token]
        months_present = sorted(bin_sample["event_time"].astype(int).unique().tolist()) if not bin_sample.empty else []

        if token == "-1":
            beta, se, pvalue, is_ref = 0.0, 0.0, np.nan, 1
        else:
            pname = binned_param_name(token)
            beta = float(result.params.get(pname, np.nan))
            se = float(bse.get(pname, np.nan))
            pvalue = float(pvalues.get(pname, np.nan))
            is_ref = 0

        rows.append({
            "family": "binned",
            "series_id": series_id,
            "series_display": SERIES_DISPLAY[series_id],
            "is_main_series": int(series_id in MAIN_SERIES),
            "reform": reform,
            "reform_display": REFORMS[reform]["display"],
            "spec_name": spec_name,
            "bin_key": str(bin_key),
            "token": token,
            "bin_label": label,
            "a_bin": int(a),
            "b_bin": int(b),
            "is_reference": is_ref,
            "n_months_observed": int(len(months_present)),
            "a_used": int(min(months_present)) if months_present else np.nan,
            "b_used": int(max(months_present)) if months_present else np.nan,
            "beta": beta,
            "se": se,
            "pvalue": pvalue,
            "n_obs": int(result.nobs),
            "r2": float(result.rsquared),
        })
    return pd.DataFrame(rows)


def binned_window_summary(result, sample: pd.DataFrame, series_id: str, reform: str, spec_name: str) -> pd.DataFrame:
    rows = []
    bin_counts: Dict[str, int] = {}
    for token, g in sample.groupby("binned_token"):
        bin_counts[str(token)] = int(g["event_time"].nunique())

    weights_by_window: Dict[str, Dict[str, float]] = {
        "avg_pre": {
            binned_param_name("pre_12_7"): float(bin_counts.get("pre_12_7", 0)),
            binned_param_name("pre_6_2"): float(bin_counts.get("pre_6_2", 0)),
        },
        "post_1_6": {binned_param_name("post_1_6"): 1.0},
        "post_7_12": {binned_param_name("post_7_12"): 1.0},
        "post_13_tail": {binned_param_name("post_13_tail"): 1.0},
        "post_full": {
            binned_param_name("post_1_6"): float(bin_counts.get("post_1_6", 0)),
            binned_param_name("post_7_12"): float(bin_counts.get("post_7_12", 0)),
            binned_param_name("post_13_tail"): float(bin_counts.get("post_13_tail", 0)),
        },
    }

    req_months_map = {
        window_key: [m for m in range(a_req, b_req + 1) if m != -1]
        for window_key, a_req, b_req, _label in REFORMS[reform]["windows"]
    }
    obs_months = sorted(set(sample["event_time"].astype(int).tolist()))

    months_map = {
        "avg_pre": [m for m in obs_months if -12 <= m <= -2],
        "post_1_6": [m for m in obs_months if 1 <= m <= 6],
        "post_7_12": [m for m in obs_months if 7 <= m <= 12],
        "post_13_tail": [m for m in obs_months if m >= 13],
        "post_full": [m for m in obs_months if m >= 1],
    }

    for window_key, a_req, b_req, label in REFORMS[reform]["windows"]:
        used_months = sorted(set(months_map[window_key]))
        required_months = req_months_map[window_key]
        full_support = int(set(required_months).issubset(set(used_months)))

        raw_weights = weights_by_window[window_key].copy()
        kept = {k: v for k, v in raw_weights.items() if np.isfinite(v) and v > 0}
        total = float(sum(kept.values()))
        weights = {k: v / total for k, v in kept.items()} if (total > 0 and full_support == 1) else {}

        beta, se, pvalue = linear_combo_from_params(result, weights)
        if full_support == 0:
            beta, se, pvalue = np.nan, np.nan, np.nan

        rows.append({
            "family": "binned",
            "series_id": series_id,
            "series_display": SERIES_DISPLAY[series_id],
            "is_main_series": int(series_id in MAIN_SERIES),
            "reform": reform,
            "reform_display": REFORMS[reform]["display"],
            "spec_name": spec_name,
            "window_key": window_key,
            "window_label": label,
            "a_req": int(a_req),
            "b_req": int(b_req),
            "a_used": int(min(used_months)) if used_months else np.nan,
            "b_used": int(max(used_months)) if used_months else np.nan,
            "n_months_used": int(len(used_months)),
            "supported_full_window": int(full_support),
            "beta": beta,
            "se": se,
            "pvalue": pvalue,
            "n_obs": int(result.nobs),
            "r2": float(result.rsquared),
        })
    return pd.DataFrame(rows)


def binned_pretrend_test(result, sample: pd.DataFrame, series_id: str, reform: str, spec_name: str) -> pd.DataFrame:
    support_pre_12_7 = set(range(-12, -6)).issubset(set(sample["event_time"].astype(int).tolist()))
    support_pre_6_2 = set(range(-6, -1)).issubset(set(sample["event_time"].astype(int).tolist()))

    tokens = []
    if support_pre_12_7 and (sample["binned_token"] == "pre_12_7").any():
        tokens.append(binned_param_name("pre_12_7"))
    if support_pre_6_2 and (sample["binned_token"] == "pre_6_2").any():
        tokens.append(binned_param_name("pre_6_2"))

    supported_full_test = int(len(tokens) == 2)
    f_val, p_val, df_num, df_denom = safe_wald_test(result, tokens) if supported_full_test == 1 else (np.nan, np.nan, np.nan, np.nan)

    return pd.DataFrame([{
        "family": "binned",
        "series_id": series_id,
        "series_display": SERIES_DISPLAY[series_id],
        "is_main_series": int(series_id in MAIN_SERIES),
        "reform": reform,
        "reform_display": REFORMS[reform]["display"],
        "spec_name": spec_name,
        "test_name": "joint_lead_bins",
        "n_terms_tested": int(len(tokens)),
        "supported_full_test": supported_full_test,
        "lead_min": -12 if tokens else np.nan,
        "lead_max": -2 if tokens else np.nan,
        "f_stat": f_val,
        "pvalue": p_val,
        "df_num": df_num,
        "df_denom": df_denom,
        "n_obs": int(result.nobs),
        "r2": float(result.rsquared),
    }])


# -----------------------------
# Sample overview and table helpers
# -----------------------------


def build_sample_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for series_id, g in df.groupby("series_id"):
        g = g.sort_values(["year", "month"]).copy()
        start_y, start_m = int(g["year"].iloc[0]), int(g["month"].iloc[0])
        end_y, end_m = int(g["year"].iloc[-1]), int(g["month"].iloc[-1])
        for reform, info in REFORMS.items():
            tmp = prepare_reform_sample(g, reform)
            et = tmp["event_time"].astype(int).tolist()
            rows.append({
                "series_id": series_id,
                "series_display": SERIES_DISPLAY[series_id],
                "is_main_series": int(series_id in MAIN_SERIES),
                "start_year": start_y,
                "start_month": start_m,
                "end_year": end_y,
                "end_month": end_m,
                "n_obs_total": int(len(g)),
                "reform": reform,
                "reform_display": info["display"],
                "support_min": int(info["support_min"]),
                "support_max": int(info["support_max"]),
                "n_obs_in_support": int(len(tmp)),
                "event_time_min_observed": int(min(et)) if et else np.nan,
                "event_time_max_observed": int(max(et)) if et else np.nan,
                "supports_avg_pre_full": int(set(range(-12, -1)).difference({-1}).issubset(set(et))),
                "supports_post_1_6_full": int(set(range(1, 7)).issubset(set(et))),
                "supports_post_7_12_full": int(set(range(7, 13)).issubset(set(et))),
                "supports_post_13_tail_full": int(set(range(13, int(info['support_max']) + 1)).issubset(set(et))),
                "supports_post_full_full": int(set(range(1, int(info['support_max']) + 1)).issubset(set(et))),
            })
    return pd.DataFrame(rows)


def build_table_cells(windows_df: pd.DataFrame, pre_df: pd.DataFrame, series_keep: List[str], table_group: str) -> pd.DataFrame:
    win = windows_df.loc[(windows_df["family"] == "binned") & (windows_df["series_id"].isin(series_keep))].copy()
    pre = pre_df.loc[(pre_df["family"] == "binned") & (pre_df["series_id"].isin(series_keep))].copy()
    if win.empty:
        return pd.DataFrame()

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
        "post_13_tail": 4,
        "post_full": 5,
        "pretrend_p": 6,
        "n_obs": 7,
        "r2": 8,
    }

    rows = []
    for _, r in win.iterrows():
        col_key = f"{r['series_id']}__{r['spec_name']}"
        col_order = (series_order.get(r["series_id"], 99) - 1) * 2 + spec_order.get(r["spec_name"], 99)
        value_display = fmt_estimate(r["beta"], r["se"], r["pvalue"]) if int(r.get("supported_full_window", 0)) == 1 else ""
        rows.append({
            "table_group": table_group,
            "reform": r["reform"],
            "reform_order": reform_order.get(r["reform"], 99),
            "row_key": r["window_key"],
            "row_label": r["window_label"],
            "row_order": row_order.get(r["window_key"], 99),
            "row_type": "estimate",
            "series_id": r["series_id"],
            "series_display": r["series_display"],
            "spec_name": r["spec_name"],
            "spec_display": SPEC_LABELS[r["spec_name"]],
            "column_key": col_key,
            "column_label": f"{r['series_display']} — {SPEC_LABELS[r['spec_name']]}",
            "column_order": col_order,
            "supported_full_window": int(r.get("supported_full_window", 0)),
            "beta": r["beta"],
            "se": r["se"],
            "pvalue": r["pvalue"],
            "n_obs": r["n_obs"],
            "r2": r["r2"],
            "a_used": r["a_used"],
            "b_used": r["b_used"],
            "n_months_used": r["n_months_used"],
            "value_display": value_display,
        })

    post_full = win.loc[win["window_key"] == "post_full"].copy()
    for _, r in pre.iterrows():
        col_key = f"{r['series_id']}__{r['spec_name']}"
        col_order = (series_order.get(r["series_id"], 99) - 1) * 2 + spec_order.get(r["spec_name"], 99)
        value_display = fmt_num(r["pvalue"], 3) if int(r.get("supported_full_test", 0)) == 1 else ""
        rows.append({
            "table_group": table_group,
            "reform": r["reform"],
            "reform_order": reform_order.get(r["reform"], 99),
            "row_key": "pretrend_p",
            "row_label": "Pretrend test p-value",
            "row_order": row_order["pretrend_p"],
            "row_type": "scalar",
            "series_id": r["series_id"],
            "series_display": r["series_display"],
            "spec_name": r["spec_name"],
            "spec_display": SPEC_LABELS[r["spec_name"]],
            "column_key": col_key,
            "column_label": f"{r['series_display']} — {SPEC_LABELS[r['spec_name']]}",
            "column_order": col_order,
            "supported_full_window": int(r.get("supported_full_test", 0)),
            "beta": np.nan,
            "se": np.nan,
            "pvalue": r["pvalue"],
            "n_obs": r["n_obs"],
            "r2": r["r2"],
            "a_used": np.nan,
            "b_used": np.nan,
            "n_months_used": r["n_terms_tested"],
            "value_display": value_display,
        })

    for _, r in post_full.iterrows():
        col_key = f"{r['series_id']}__{r['spec_name']}"
        col_order = (series_order.get(r["series_id"], 99) - 1) * 2 + spec_order.get(r["spec_name"], 99)
        rows.append({
            "table_group": table_group,
            "reform": r["reform"],
            "reform_order": reform_order.get(r["reform"], 99),
            "row_key": "n_obs",
            "row_label": "Observations",
            "row_order": row_order["n_obs"],
            "row_type": "scalar",
            "series_id": r["series_id"],
            "series_display": r["series_display"],
            "spec_name": r["spec_name"],
            "spec_display": SPEC_LABELS[r["spec_name"]],
            "column_key": col_key,
            "column_label": f"{r['series_display']} — {SPEC_LABELS[r['spec_name']]}",
            "column_order": col_order,
            "supported_full_window": int(r.get("supported_full_window", 0)),
            "beta": np.nan,
            "se": np.nan,
            "pvalue": np.nan,
            "n_obs": r["n_obs"],
            "r2": r["r2"],
            "a_used": np.nan,
            "b_used": np.nan,
            "n_months_used": np.nan,
            "value_display": fmt_intlike(r["n_obs"]),
        })
        rows.append({
            "table_group": table_group,
            "reform": r["reform"],
            "reform_order": reform_order.get(r["reform"], 99),
            "row_key": "r2",
            "row_label": "R-squared",
            "row_order": row_order["r2"],
            "row_type": "scalar",
            "series_id": r["series_id"],
            "series_display": r["series_display"],
            "spec_name": r["spec_name"],
            "spec_display": SPEC_LABELS[r["spec_name"]],
            "column_key": col_key,
            "column_label": f"{r['series_display']} — {SPEC_LABELS[r['spec_name']]}",
            "column_order": col_order,
            "supported_full_window": int(r.get("supported_full_window", 0)),
            "beta": np.nan,
            "se": np.nan,
            "pvalue": np.nan,
            "n_obs": r["n_obs"],
            "r2": r["r2"],
            "a_used": np.nan,
            "b_used": np.nan,
            "n_months_used": np.nan,
            "value_display": fmt_num(r["r2"], 3),
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["reform_order", "row_order", "column_order"]).reset_index(drop=True)


# -----------------------------
# Main routine
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Model 1B v7: exact plot dynamics + binned inference with dynamic overlap controls")
    parser.add_argument("--kl", default=None, help="Path to KL_Panel_monthly.tsv")
    parser.add_argument("--outdir", default=None, help="Optional output directory")
    parser.add_argument("--hac-lags", type=int, default=6, help="Newey-West maxlags")
    args = parser.parse_args()

    thesis_root = find_thesis_root()
    kl_path = Path(args.kl) if args.kl else thesis_root / "Data" / "KL" / "KL_Panel_monthly.tsv"
    if not kl_path.is_absolute():
        kl_path = thesis_root / kl_path

    output_dir = Path(args.outdir) if args.outdir else thesis_root / "Design" / "Output (new)" / "Model_1B_v7"
    if not output_dir.is_absolute():
        output_dir = thesis_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_outputs(output_dir)

    df = load_kl_panel(kl_path)

    exact_rows_all: List[pd.DataFrame] = []
    binned_rows_all: List[pd.DataFrame] = []
    binned_windows_all: List[pd.DataFrame] = []
    binned_pre_all: List[pd.DataFrame] = []

    for series_id in ALL_SERIES:
        df_series = df.loc[df["series_id"] == series_id].copy().sort_values(["year", "month"]).reset_index(drop=True)
        print(f"\n=== Series: {series_id} ({len(df_series)} rows) ===")

        for reform in ["competition", "privatization"]:
            print(f"  Reform: {reform}")
            sample = prepare_reform_sample(df_series, reform)
            sample["exact_token"] = sample["event_time"].astype(int).map(exact_token)
            sample["binned_token"] = sample["event_time"].astype(int).map(lambda m: assign_binned_token(m, reform))

            # Exact family: plot / descriptive only
            for spec_name in ["baseline", "ctrl_trend"]:
                print(f"    Exact spec: {spec_name}")
                formula = build_formula("exact", reform, spec_name)
                result = fit_hac_ols(formula, sample, args.hac_lags)
                exact_rows_all.append(extract_exact_rows(result, sample, series_id, reform, spec_name))

            # Binned family: main inference
            for spec_name in ["baseline", "ctrl_trend"]:
                print(f"    Binned spec: {spec_name}")
                formula = build_formula("binned", reform, spec_name)
                result = fit_hac_ols(formula, sample, args.hac_lags)
                binned_rows_all.append(extract_binned_rows(result, sample, series_id, reform, spec_name))
                binned_windows_all.append(binned_window_summary(result, sample, series_id, reform, spec_name))
                binned_pre_all.append(binned_pretrend_test(result, sample, series_id, reform, spec_name))

    exact_rows = pd.concat(exact_rows_all, ignore_index=True) if exact_rows_all else pd.DataFrame()
    binned_rows = pd.concat(binned_rows_all, ignore_index=True) if binned_rows_all else pd.DataFrame()
    binned_windows = pd.concat(binned_windows_all, ignore_index=True) if binned_windows_all else pd.DataFrame()
    binned_pre = pd.concat(binned_pre_all, ignore_index=True) if binned_pre_all else pd.DataFrame()
    sample_df = build_sample_overview(df)

    plot_helper = exact_rows.loc[exact_rows["event_time"] != -1].copy().sort_values([
        "series_id", "reform", "spec_name", "event_time"
    ])

    main_table = build_table_cells(binned_windows, binned_pre, MAIN_SERIES, "main")
    bayport_table = build_table_cells(binned_windows, binned_pre, SUPPLEMENTARY_SERIES, "bayport_appendix")

    paths = {
        "exact_dynamic": output_dir / "model1b_exact_dynamic_betas_all.tsv",
        "binned_dynamic": output_dir / "model1b_binned_betas_all.tsv",
        "binned_windows": output_dir / "model1b_binned_window_betas_summary.tsv",
        "binned_pretrend": output_dir / "model1b_binned_pretrend_tests.tsv",
        "sample_overview": output_dir / "model1b_sample_overview.tsv",
        "plot_helper": output_dir / "model1b_plot_helper.tsv",
        "main_table": output_dir / "model1b_table_cells_main.tsv",
        "bayport_table": output_dir / "model1b_table_cells_bayport_appendix.tsv",
        "manifest": output_dir / "model1b_manifest.json",
    }

    exact_rows.to_csv(paths["exact_dynamic"], sep="\t", index=False)
    binned_rows.to_csv(paths["binned_dynamic"], sep="\t", index=False)
    binned_windows.to_csv(paths["binned_windows"], sep="\t", index=False)
    binned_pre.to_csv(paths["binned_pretrend"], sep="\t", index=False)
    sample_df.to_csv(paths["sample_overview"], sep="\t", index=False)
    plot_helper.to_csv(paths["plot_helper"], sep="\t", index=False)
    main_table.to_csv(paths["main_table"], sep="\t", index=False)
    bayport_table.to_csv(paths["bayport_table"], sep="\t", index=False)

    manifest = {
        "script": "Model_1B_v7.py",
        "kl_input": str(kl_path),
        "output_dir": str(output_dir),
        "hac_lags": int(args.hac_lags),
        "main_series": MAIN_SERIES,
        "supplementary_series": SUPPLEMENTARY_SERIES,
        "reforms": REFORMS,
        "specifications": {
            "baseline": "event terms + dynamic overlap control",
            "ctrl_trend": "event terms + dynamic overlap control + t_linear + covid_shock + war_shock",
        },
        "families": {
            "exact": "exact-month event-study on reform-specific support; plot/descriptive use only",
            "binned": "relaxed binned event-study used for the main tables",
        },
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    with open(paths["manifest"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=json_default)

    print("\n=== Model_1B_v7: done ===")
    print(f"Wrote exact dynamic betas : {paths['exact_dynamic']}")
    print(f"Wrote binned betas        : {paths['binned_dynamic']}")
    print(f"Wrote binned windows      : {paths['binned_windows']}")
    print(f"Wrote binned pretrends    : {paths['binned_pretrend']}")
    print(f"Wrote sample overview     : {paths['sample_overview']}")
    print(f"Wrote plot helper         : {paths['plot_helper']}")
    print(f"Wrote main table          : {paths['main_table']}")
    print(f"Wrote Bayport table       : {paths['bayport_table']}")
    print(f"Wrote manifest            : {paths['manifest']}")


if __name__ == "__main__":
    main()
