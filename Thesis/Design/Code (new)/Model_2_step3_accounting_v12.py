#!/usr/bin/env python3
"""
Model_2_step3_accounting_v12.py

Build the enhanced Model 2 accounting layer against the intended final
Model 1A v8.2 and Model 1B v8 files.

v10 enhancements
----------------
1. Carries approximate uncertainty for derived dC objects built from Model 1B
   summary bins, using independence-based propagation.
2. Propagates approximate uncertainty to CD = eta*dC and share explained,
   using a simple delta-method style approximation.
3. Makes aggregate-regression fallback explicit in row metadata instead of
   letting those rows look identical to preferred-manual rows.
4. Writes richer provenance columns for appendix/debugging.
5. Writes compact diagnostics tables for all rows and for the main rows.
6. Writes an LP-family / horizon coverage table so asymmetric competition
   comparison is explicit rather than hidden.
7. Writes the preferred-row rule to JSON so the editorial selection rule is
   transparent and reproducible.

Notes
-----
- Manual eta remains the preferred default when available.
- No manual aggregate eta is invented here. Aggregate rows fall back to the
  regression eta and are labeled as such.
- Uncertainty propagation is approximate and assumes independence across KL
  component bins and between eta and dC.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------

def find_thesis_root(start: Optional[Path] = None) -> Path:
    here = start or Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "Data").exists() and (p / "Design").exists():
            return p
    raise FileNotFoundError("Could not locate thesis root.")


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def read_tsv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t")


def first_present(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols_l = {str(c).lower().strip(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
    return None


def normalize_string_col(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or not df.columns.duplicated().any():
        return df
    out = pd.DataFrame(index=df.index)
    seen: List[str] = []
    for name in df.columns:
        if name in seen:
            continue
        seen.append(name)
        cols = df.loc[:, df.columns == name]
        s = cols.iloc[:, 0].copy()
        for j in range(1, cols.shape[1]):
            s = s.combine_first(cols.iloc[:, j])
        out[name] = s
    return out


def clean_token(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower().replace("_", " ").replace("--", "-").replace("  ", " ")


def any_match(value: object, aliases: Sequence[str]) -> bool:
    v = clean_token(value)
    return v in {clean_token(a) for a in aliases}


def contains_any(text: str, tokens: Sequence[str]) -> bool:
    return any(tok in text for tok in tokens)


def format_horizon(m_start: float, m_end: float) -> str:
    if pd.isna(m_start) or pd.isna(m_end):
        return ""
    return f"[{int(m_start)},{int(m_end)}]"


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_pvalue(beta: float, se: float) -> float:
    if not (np.isfinite(beta) and np.isfinite(se) and se > 0):
        return np.nan
    z = abs(beta / se)
    return max(0.0, min(1.0, 2.0 * (1.0 - norm_cdf(z))))


def ci_from_beta_se(beta: float, se: float, level: float = 0.95) -> Tuple[float, float]:
    if not (np.isfinite(beta) and np.isfinite(se) and se >= 0):
        return (np.nan, np.nan)
    z = 1.959963984540054
    return (beta - z * se, beta + z * se)


# ---------------------------------------------------------------------
# Row config
# ---------------------------------------------------------------------

REFORM_ALIASES = {
    "haifa_comp": [
        "haifa_comp", "haifa competition", "competition_haifa", "competition haifa",
        "competition", "comp_haifa",
    ],
    "haifa_priv": [
        "haifa_priv", "haifa privatization", "privatization_haifa", "privatization haifa",
        "privatization", "priv_haifa",
    ],
}


@dataclass(frozen=True)
class RowConfig:
    row_key: str
    row_label: str
    reform: str
    lp_entity: str
    kl_entity: str
    lp_target_aliases: Sequence[str]
    kl_target_aliases: Sequence[str]
    preferred_eta_entity: str
    preferred_horizon: Tuple[int, int]
    preferred_lp_family_order: Sequence[str]
    preferred_lp_spec_order: Sequence[str]
    preferred_kl_spec_order: Sequence[str]
    preferred_kl_variant_order: Sequence[str]


ROW_CONFIGS: List[RowConfig] = [
    RowConfig(
        row_key="competition_legacy",
        row_label="Haifa competition - Legacy",
        reform="haifa_comp",
        lp_entity="Haifa--Legacy",
        kl_entity="Haifa--Legacy",
        lp_target_aliases=[
            "Haifa--Legacy", "Haifa-Legacy", "Haifa Legacy", "Haifa_Legacy",
            "Haifa-Legacy terminal", "Haifa legacy terminal",
        ],
        kl_target_aliases=[
            "Haifa--Legacy", "Haifa-Legacy", "Haifa Legacy", "Haifa_Legacy",
            "Haifa_Legacy_KL", "Haifa Legacy KL", "Haifa-Legacy K/L", "Haifa legacy K/L",
        ],
        preferred_eta_entity="Haifa--Legacy",
        preferred_horizon=(1, 13),
        preferred_lp_family_order=("Conventional DiD", "NYT"),
        preferred_lp_spec_order=("Baseline", "+Tr"),
        preferred_kl_spec_order=("Baseline", "Controls+Trend"),
        preferred_kl_variant_order=("SummaryDerived",),
    ),
    RowConfig(
        row_key="competition_aggregate",
        row_label="Haifa competition - Aggregate",
        reform="haifa_comp",
        lp_entity="Haifa-Aggregate",
        kl_entity="Haifa port cluster",
        lp_target_aliases=[
            "Haifa-Aggregate", "Haifa Aggregate", "Haifa_Aggregate",
            "Haifa port cluster", "Haifa port", "Haifa", "Haifa aggregate", "Haifa_port_Q", "Haifa_port",
        ],
        kl_target_aliases=[
            "Haifa port cluster", "Haifa port", "Haifa", "Haifa aggregate",
            "Haifa_port_KL", "Haifa port KL", "Haifa_port",
        ],
        preferred_eta_entity="Haifa port cluster",
        preferred_horizon=(1, 13),
        preferred_lp_family_order=("Conventional DiD", "NYT"),
        preferred_lp_spec_order=("Baseline", "+Tr"),
        preferred_kl_spec_order=("Baseline", "Controls+Trend"),
        preferred_kl_variant_order=("SummaryDerived",),
    ),
    RowConfig(
        row_key="privatization_legacy",
        row_label="Haifa privatization - Legacy",
        reform="haifa_priv",
        lp_entity="Haifa--Legacy",
        kl_entity="Haifa--Legacy",
        lp_target_aliases=[
            "Haifa--Legacy", "Haifa-Legacy", "Haifa Legacy", "Haifa_Legacy",
            "Haifa-Legacy terminal", "Haifa legacy terminal",
        ],
        kl_target_aliases=[
            "Haifa--Legacy", "Haifa-Legacy", "Haifa Legacy", "Haifa_Legacy",
            "Haifa_Legacy_KL", "Haifa Legacy KL", "Haifa-Legacy K/L", "Haifa legacy K/L",
        ],
        preferred_eta_entity="Haifa--Legacy",
        preferred_horizon=(1, 7),
        preferred_lp_family_order=("NYT", "Conventional DiD"),
        preferred_lp_spec_order=("Baseline", "+Tr"),
        preferred_kl_spec_order=("Baseline", "Controls+Trend"),
        preferred_kl_variant_order=("SummaryDerived",),
    ),
    RowConfig(
        row_key="privatization_aggregate",
        row_label="Haifa privatization - Aggregate",
        reform="haifa_priv",
        lp_entity="Haifa-Aggregate",
        kl_entity="Haifa port cluster",
        lp_target_aliases=[
            "Haifa-Aggregate", "Haifa Aggregate", "Haifa_Aggregate",
            "Haifa port cluster", "Haifa port", "Haifa", "Haifa aggregate", "Haifa_port_Q", "Haifa_port",
        ],
        kl_target_aliases=[
            "Haifa port cluster", "Haifa port", "Haifa", "Haifa aggregate",
            "Haifa_port_KL", "Haifa port KL", "Haifa_port",
        ],
        preferred_eta_entity="Haifa port cluster",
        preferred_horizon=(1, 7),
        preferred_lp_family_order=("NYT", "Conventional DiD"),
        preferred_lp_spec_order=("Baseline", "+Tr"),
        preferred_kl_spec_order=("Baseline", "Controls+Trend"),
        preferred_kl_variant_order=("SummaryDerived",),
    ),
]


# ---------------------------------------------------------------------
# Schema normalization
# ---------------------------------------------------------------------

WINDOW_RE_1 = re.compile(r"months\s*([\-0-9]+)\s*[-–]\s*([\-0-9]+)", flags=re.I)
WINDOW_RE_2 = re.compile(r"\[\s*([\-0-9]+)\s*,\s*([\-0-9]+)\s*\]")


def parse_window_bounds(text: object) -> Tuple[float, float]:
    if pd.isna(text):
        return (np.nan, np.nan)
    s = str(text)
    m = WINDOW_RE_1.search(s)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    m = WINDOW_RE_2.search(s)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return (np.nan, np.nan)


def normalize_window_df(df: Optional[pd.DataFrame], kind: str) -> Optional[pd.DataFrame]:
    if df is None:
        return None

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    ren = {}
    for cand_list, dst in [
        (["reform", "reform_key"], "reform"),
        (["target", "entity", "unit", "series_id", "name", "label"], "target"),
        (["design", "estimator_family", "model_type", "family"], "design"),
        (["spec_name", "fe_type", "spec", "spec_key", "specification", "column_name"], "spec_name"),
        (["window_name", "window", "horizon_key", "horizon", "window_label", "post_window"], "window_name"),
        (["m_start", "start_m", "window_start", "event_start", "a"], "m_start"),
        (["m_end", "end_m", "window_end", "event_end", "b"], "m_end"),
        (["beta_hat", "beta", "coef", "coefficient", "estimate", "avg_beta"], "beta_hat"),
        (["se", "stderr", "std_err", "beta_se", "avg_se"], "se"),
        (["pvalue", "p_value", "p"], "pvalue"),
        (["n_obs", "N", "n", "obs"], "N"),
        (["r2", "R2", "within_r2", "rsq"], "R2"),
    ]:
        c = first_present(out.columns, cand_list)
        if c and c != dst:
            ren[c] = dst
    out = out.rename(columns=ren)
    out = collapse_duplicate_columns(out)

    required = ["reform", "target", "beta_hat"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Window file missing required columns after normalization: {missing}")

    for c in ["reform", "target", "design", "spec_name", "window_name"]:
        if c in out.columns:
            out[c] = normalize_string_col(out[c])

    for c in ["m_start", "m_end", "beta_hat", "se", "pvalue", "N", "R2"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "design" not in out.columns:
        out["design"] = ""
    if "spec_name" not in out.columns:
        out["spec_name"] = ""
    if "window_name" not in out.columns:
        out["window_name"] = ""
    if "m_start" not in out.columns:
        out["m_start"] = np.nan
    if "m_end" not in out.columns:
        out["m_end"] = np.nan

    if kind == "m1b":
        missing_bounds = out["m_start"].isna() | out["m_end"].isna()
        if missing_bounds.any():
            parsed = out.loc[missing_bounds, "window_name"].apply(parse_window_bounds)
            out.loc[missing_bounds, "m_start"] = [a for a, _ in parsed]
            out.loc[missing_bounds, "m_end"] = [b for _, b in parsed]

    return out


def normalize_eta_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    ren = {}
    for cand_list, dst in [
        (["entity"], "entity"),
        (["eta_family", "family"], "eta_family"),
        (["eta_source", "source"], "eta_source"),
        (["eta_role", "role"], "eta_role"),
        (["eta_usage_class", "usage_class"], "eta_usage_class"),
        (["eta"], "eta"),
        (["eta_se", "se_eta", "se"], "eta_se"),
        (["spec", "spec_name"], "spec"),
        (["preferred_flag", "preferred"], "preferred_flag"),
        (["status"], "status"),
        (["reason", "notes"], "reason"),
    ]:
        c = first_present(out.columns, cand_list)
        if c and c != dst:
            ren[c] = dst
    out = out.rename(columns=ren)

    for c in ["entity", "eta_family", "eta_source", "eta_role", "eta_usage_class", "spec", "status", "reason"]:
        if c in out.columns:
            out[c] = normalize_string_col(out[c])

    for c in ["eta", "eta_se", "preferred_flag", "delta", "pvalue", "N", "R2"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "preferred_flag" not in out.columns:
        out["preferred_flag"] = 0
    if "status" not in out.columns:
        out["status"] = "ok"
    if "eta_role" not in out.columns:
        out["eta_role"] = ""
    if "eta_usage_class" not in out.columns:
        out["eta_usage_class"] = ""
    if "spec" not in out.columns:
        out["spec"] = ""
    return out


# ---------------------------------------------------------------------
# Semantic categorization
# ---------------------------------------------------------------------

def infer_lp_family(row: pd.Series) -> str:
    text = " ".join([clean_token(row.get(c, "")) for c in ["design", "spec_name", "window_name"]])
    if contains_any(text, ["nyt", "not yet treated", "sun", "abraham"]):
        return "NYT"
    if contains_any(text, ["conventional", "did", "twfe", "full post", "post year"]):
        return "Conventional DiD"
    return "Unknown"


def infer_lp_spec(row: pd.Series) -> str:
    text = " ".join([clean_token(row.get(c, "")) for c in ["design", "spec_name"]])
    if contains_any(text, ["+tr", " trend", "trend", "porttr", "_tr", " tr"]):
        return "+Tr"
    return "Baseline"


def infer_kl_spec(row: pd.Series) -> str:
    text = " ".join([clean_token(row.get(c, "")) for c in ["design", "spec_name"]])
    if contains_any(text, ["ctrl", "controls", "control", "trend", "+tr", "_tr", " tr"]):
        return "Controls+Trend"
    return "Baseline"


# ---------------------------------------------------------------------
# Matching and uncertainty helpers
# ---------------------------------------------------------------------

def subset_for_row(df: pd.DataFrame, reform_key: str, target_aliases: Sequence[str]) -> pd.DataFrame:
    reform_mask = df["reform"].apply(lambda x: any_match(x, REFORM_ALIASES[reform_key]))
    target_mask = df["target"].apply(lambda x: any_match(x, target_aliases))
    return df.loc[reform_mask & target_mask].copy()


def pick_eta_rows(eta_df: pd.DataFrame, entity: str) -> pd.DataFrame:
    sub = eta_df[eta_df["entity"].astype(str) == entity].copy()
    return sub[sub["eta"].notna()].copy()


def manual_eta_available(eta_df: pd.DataFrame, entity: str) -> bool:
    sub = eta_df[(eta_df["entity"].astype(str) == entity) & (eta_df["eta_family"].astype(str) == "manual") & (eta_df["eta"].notna())]
    return not sub.empty


def choose_eta_entity(row_cfg: RowConfig, eta_df: pd.DataFrame) -> Optional[str]:
    preferred = row_cfg.preferred_eta_entity
    ok = eta_df[(eta_df["entity"].astype(str) == preferred) & (eta_df["eta"].notna())].copy()
    if not ok.empty:
        return preferred
    legacy_fallback = eta_df[(eta_df["entity"].astype(str) == "Haifa--Legacy") & (eta_df["eta"].notna())].copy()
    if not legacy_fallback.empty:
        return "Haifa--Legacy"
    return None


def classify_eta_interpretation(row: pd.Series, row_cfg: RowConfig, eta_df: pd.DataFrame) -> str:
    entity = str(row.get("eta_entity", ""))
    family = str(row.get("eta_family", ""))
    usage = str(row.get("eta_usage_class", ""))
    preferred = int(row.get("eta_preferred_flag", 0) or 0)

    if row_cfg.kl_entity == "Haifa port cluster" and family == "manual" and preferred == 1:
        return "preferred_manual_aggregate"
    if family == "manual" and preferred == 1:
        return "preferred_manual"
    if row_cfg.kl_entity == "Haifa port cluster" and not manual_eta_available(eta_df, "Haifa port cluster") and family == "regression":
        return "aggregate_regression_fallback"
    if family == "manual":
        return "manual_robustness"
    if entity == "Haifa port cluster" and family == "regression":
        return "aggregate_regression_fallback"
    if usage:
        return usage
    return "regression_robustness"



def compute_accounting(te: float, dc: float, eta: float) -> Tuple[float, float, float, bool, bool]:
    if not (np.isfinite(te) and np.isfinite(dc) and np.isfinite(eta)):
        return (np.nan, np.nan, np.nan, False, False)
    cd = eta * dc
    residual = te - cd
    share = cd / te if te != 0 else np.nan
    sign_consistent = bool(np.sign(te) == np.sign(cd)) if te != 0 and cd != 0 else False
    share_valid = bool(np.isfinite(share) and abs(te) > 1e-10 and sign_consistent)
    return (cd, residual, share, sign_consistent, share_valid)


def compute_cd_uncertainty(dC: float, dC_se: float, eta: float, eta_se: float) -> Tuple[float, float, float]:
    var_dc = float(dC_se ** 2) if np.isfinite(dC_se) else np.nan
    var_eta = float(eta_se ** 2) if np.isfinite(eta_se) else 0.0
    if not (np.isfinite(dC) and np.isfinite(eta)):
        return (np.nan, np.nan, np.nan)
    if not np.isfinite(var_dc) and not np.isfinite(var_eta):
        return (np.nan, np.nan, np.nan)
    if not np.isfinite(var_dc):
        var_dc = 0.0
    var_cd = (eta ** 2) * var_dc + (dC ** 2) * var_eta
    se_cd = math.sqrt(var_cd) if var_cd >= 0 else np.nan
    lo, hi = ci_from_beta_se(eta * dC, se_cd)
    return (se_cd, lo, hi)


def compute_share_uncertainty(te: float, te_se: float, cd: float, cd_se: float) -> Tuple[float, float, float]:
    if not (np.isfinite(te) and te != 0 and np.isfinite(cd)):
        return (np.nan, np.nan, np.nan)
    var_te = float(te_se ** 2) if np.isfinite(te_se) else np.nan
    var_cd = float(cd_se ** 2) if np.isfinite(cd_se) else np.nan
    if not np.isfinite(var_te) and not np.isfinite(var_cd):
        return (np.nan, np.nan, np.nan)
    if not np.isfinite(var_te):
        var_te = 0.0
    if not np.isfinite(var_cd):
        var_cd = 0.0
    # delta method for s = cd / te, ignoring covariance
    var_s = (1.0 / te) ** 2 * var_cd + ((cd / (te ** 2)) ** 2) * var_te
    se_s = math.sqrt(var_s) if var_s >= 0 else np.nan
    lo, hi = ci_from_beta_se(cd / te, se_s)
    return (se_s, lo, hi)


def find_exact_row(df: pd.DataFrame, m_start: int, m_end: int, spec_name: str) -> Optional[pd.Series]:
    sub = df[
        (pd.to_numeric(df["m_start"], errors="coerce") == float(m_start)) &
        (pd.to_numeric(df["m_end"], errors="coerce") == float(m_end)) &
        (df["spec_name"].astype(str) == spec_name)
    ].copy()
    if sub.empty:
        return None
    return sub.iloc[0]


def combine_component_rows(rows: Sequence[pd.Series], month_weights: Sequence[float], horizon: Tuple[int, int], spec_name: str, description: str) -> Dict[str, object]:
    total_w = float(sum(month_weights))
    weights = [float(w) / total_w for w in month_weights]
    betas = [float(r.get("beta_hat", np.nan)) for r in rows]
    ses = [float(r.get("se", np.nan)) for r in rows]
    pvalues = [float(r.get("pvalue", np.nan)) for r in rows]
    beta = float(sum(w * b for w, b in zip(weights, betas)))

    if all(np.isfinite(se) for se in ses):
        # independence-based approximation
        se = math.sqrt(sum((w ** 2) * (se_i ** 2) for w, se_i in zip(weights, ses)))
        pvalue = two_sided_pvalue(beta, se)
    else:
        se = np.nan
        pvalue = np.nan

    comp_windows = []
    comp_weights = []
    comp_betas = []
    comp_ses = []
    for r, mw, nw in zip(rows, month_weights, weights):
        comp_windows.append(format_horizon(float(r.get("m_start", np.nan)), float(r.get("m_end", np.nan))))
        comp_weights.append(f"{mw:g}")
        comp_betas.append(f"{float(r.get('beta_hat', np.nan)):.6g}" if np.isfinite(float(r.get('beta_hat', np.nan))) else "nan")
        comp_ses.append(f"{float(r.get('se', np.nan)):.6g}" if np.isfinite(float(r.get('se', np.nan))) else "nan")

    return {
        "m_start": int(horizon[0]),
        "m_end": int(horizon[1]),
        "beta_hat": beta,
        "se": se,
        "pvalue": pvalue,
        "N": np.nan,
        "R2": np.nan,
        "window_name": description,
        "spec_name": spec_name,
        "design": "binned",
        "kl_variant": "SummaryDerived",
        "derivation_note": "Weighted-average K/L effect from intended Model 1B summary bins; uncertainty assumes independent component bins.",
        "component_windows": "; ".join(comp_windows),
        "component_month_weights": "; ".join(comp_weights),
        "component_normalized_weights": "; ".join(f"{w:.6f}" for w in weights),
        "component_betas": "; ".join(comp_betas),
        "component_ses": "; ".join(comp_ses),
        "dC_se_method": "independence_weighted",
    }


def derive_kl_row(df: pd.DataFrame, row_key: str, spec_name: str) -> Optional[Dict[str, object]]:
    spec_df = df[df["spec_name"].astype(str) == spec_name].copy()
    if spec_df.empty:
        return None

    if row_key in ("competition_legacy", "competition_aggregate"):
        r1 = find_exact_row(spec_df, 1, 6, spec_name)
        r2 = find_exact_row(spec_df, 7, 12, spec_name)
        r3 = find_exact_row(spec_df, 13, 24, spec_name)
        if r3 is None:
            r3 = find_exact_row(spec_df, 13, 23, spec_name)
        if r1 is None or r2 is None or r3 is None:
            return None
        return combine_component_rows(
            [r1, r2, r3],
            [6, 6, 1],
            (1, 13),
            spec_name,
            "Derived from [1,6], [7,12], [13,24/23]",
        )

    if row_key in ("privatization_legacy", "privatization_aggregate"):
        r1 = find_exact_row(spec_df, 1, 6, spec_name)
        r2 = find_exact_row(spec_df, 7, 12, spec_name)
        if r1 is None or r2 is None:
            return None
        return combine_component_rows(
            [r1, r2],
            [6, 1],
            (1, 7),
            spec_name,
            "Derived from [1,6] plus month 7 from [7,12]",
        )

    return None


def preferred_score(row: pd.Series, cfg: RowConfig) -> tuple:
    def pos(value: str, ordering: Sequence[str], fallback: int = 99) -> int:
        try:
            return list(ordering).index(value)
        except ValueError:
            return fallback

    eta_usage = str(row.get("eta_interpretation_class", ""))
    eta_rank_map = {
        "preferred_manual": 0,
        "preferred_manual_aggregate": 1,
        "manual_robustness": 2,
        "aggregate_regression_fallback": 3,
        "regression_robustness": 4,
        "diagnostic": 5,
        "unavailable": 99,
    }
    te_se = pd.to_numeric(row.get("TE_se", np.nan), errors="coerce")
    dc_se = pd.to_numeric(row.get("dC_se", np.nan), errors="coerce")
    te_r2 = pd.to_numeric(row.get("TE_R2", np.nan), errors="coerce")

    horizon_pref = 0 if (int(row.get("m_start", -999)) == cfg.preferred_horizon[0] and int(row.get("m_end", -999)) == cfg.preferred_horizon[1]) else 1
    return (
        horizon_pref,
        pos(str(row.get("lp_family", "")), cfg.preferred_lp_family_order),
        pos(str(row.get("lp_spec", "")), cfg.preferred_lp_spec_order),
        pos(str(row.get("kl_spec", "")), cfg.preferred_kl_spec_order),
        pos(str(row.get("kl_variant", "")), cfg.preferred_kl_variant_order),
        eta_rank_map.get(eta_usage, 50),
        0 if bool(row.get("sign_consistent", False)) else 1,
        0 if bool(row.get("share_valid", False)) else 1,
        float(te_se) if np.isfinite(te_se) else 1e9,
        float(dc_se) if np.isfinite(dc_se) else 1e9,
        -float(te_r2) if np.isfinite(te_r2) else 0.0,
    )


# ---------------------------------------------------------------------
# Audits and diagnostics
# ---------------------------------------------------------------------

def build_audit(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["kind", "reform", "target", "design", "spec_name", "window_name", "m_start", "m_end", "n_rows"])
    out = (
        df.groupby(["reform", "target", "design", "spec_name", "window_name", "m_start", "m_end"], dropna=False)
          .size()
          .reset_index(name="n_rows")
    )
    out.insert(0, "kind", kind)
    return out


def build_family_coverage(lp_df: pd.DataFrame, kl_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for cfg in ROW_CONFIGS:
        lp_sub = subset_for_row(lp_df, cfg.reform, cfg.lp_target_aliases).copy()
        lp_sub["lp_family"] = lp_sub.apply(infer_lp_family, axis=1)
        for fam, fam_sub in lp_sub.groupby("lp_family", dropna=False):
            horizons = sorted({format_horizon(a, b) for a, b in fam_sub[["m_start", "m_end"]].itertuples(index=False, name=None) if np.isfinite(a) and np.isfinite(b)})
            rows.append({
                "row_key": cfg.row_key,
                "row_label": cfg.row_label,
                "reform": cfg.reform,
                "lp_family": fam,
                "available_lp_horizons": "; ".join(horizons),
                "preferred_horizon": format_horizon(*cfg.preferred_horizon),
                "preferred_horizon_available_in_lp": format_horizon(*cfg.preferred_horizon) in horizons,
                "kl_summary_derivable_at_preferred_horizon": derive_kl_row(subset_for_row(kl_df, cfg.reform, cfg.kl_target_aliases), cfg.row_key, "baseline") is not None,
                "note": "Competition NYT is asymmetric when Model 1A only provides shorter positive post windows than the preferred accounting horizon.",
            })
    return pd.DataFrame(rows)


def build_diagnostics_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "row_key", "row_label", "horizon", "lp_family", "lp_spec", "kl_spec", "eta_interpretation_class",
            "TE", "TE_se", "TE_ci", "TE_R2", "dC", "dC_se", "dC_ci", "eta", "eta_se",
            "CD", "CD_se", "CD_ci", "share_explained", "share_se", "share_ci",
            "sign_consistent", "share_valid", "kl_component_windows", "kl_component_month_weights", "warning_flags"
        ])
    rows: List[Dict[str, object]] = []
    for _, r in df.iterrows():
        warnings = []
        if str(r.get("eta_interpretation_class", "")) == "aggregate_regression_fallback":
            warnings.append("aggregate_eta_fallback")
        if not bool(r.get("share_valid", False)):
            warnings.append("share_not_mechanically_clean")
        if not np.isfinite(pd.to_numeric(r.get("dC_se", np.nan), errors="coerce")):
            warnings.append("missing_dC_se")
        rows.append({
            "row_key": r.get("row_key", ""),
            "row_label": r.get("row_label", ""),
            "horizon": r.get("horizon", ""),
            "lp_family": r.get("lp_family", ""),
            "lp_spec": r.get("lp_spec", ""),
            "kl_spec": r.get("kl_spec", ""),
            "eta_interpretation_class": r.get("eta_interpretation_class", ""),
            "TE": r.get("TE", np.nan),
            "TE_se": r.get("TE_se", np.nan),
            "TE_ci": f"[{r.get('TE_ci_lo', np.nan):.3f},{r.get('TE_ci_hi', np.nan):.3f}]" if np.isfinite(pd.to_numeric(r.get('TE_ci_lo', np.nan), errors='coerce')) else "",
            "TE_R2": r.get("TE_R2", np.nan),
            "dC": r.get("dC", np.nan),
            "dC_se": r.get("dC_se", np.nan),
            "dC_ci": f"[{r.get('dC_ci_lo', np.nan):.3f},{r.get('dC_ci_hi', np.nan):.3f}]" if np.isfinite(pd.to_numeric(r.get('dC_ci_lo', np.nan), errors='coerce')) else "",
            "eta": r.get("eta", np.nan),
            "eta_se": r.get("eta_se", np.nan),
            "CD": r.get("CD", np.nan),
            "CD_se": r.get("CD_se", np.nan),
            "CD_ci": f"[{r.get('CD_ci_lo', np.nan):.3f},{r.get('CD_ci_hi', np.nan):.3f}]" if np.isfinite(pd.to_numeric(r.get('CD_ci_lo', np.nan), errors='coerce')) else "",
            "share_explained": r.get("share_explained", np.nan),
            "share_se": r.get("share_se", np.nan),
            "share_ci": f"[{r.get('share_ci_lo', np.nan):.3f},{r.get('share_ci_hi', np.nan):.3f}]" if np.isfinite(pd.to_numeric(r.get('share_ci_lo', np.nan), errors='coerce')) else "",
            "sign_consistent": r.get("sign_consistent", False),
            "share_valid": r.get("share_valid", False),
            "kl_component_windows": r.get("kl_component_windows", ""),
            "kl_component_month_weights": r.get("kl_component_month_weights", ""),
            "warning_flags": "; ".join(warnings),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------

def build_long_rows(lp_df: pd.DataFrame, kl_df: pd.DataFrame, eta_df: pd.DataFrame, lp_source_map: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ok_rows: List[Dict[str, object]] = []
    unavailable_rows: List[Dict[str, object]] = []

    for cfg in ROW_CONFIGS:
        lp_sub = subset_for_row(lp_df, cfg.reform, cfg.lp_target_aliases)
        kl_sub = subset_for_row(kl_df, cfg.reform, cfg.kl_target_aliases)

        if lp_sub.empty:
            unavailable_rows.append({
                "row_key": cfg.row_key,
                "row_label": cfg.row_label,
                "status": "unavailable",
                "reason": "No matching Model 1A rows found for this row config.",
            })
            continue
        if kl_sub.empty:
            unavailable_rows.append({
                "row_key": cfg.row_key,
                "row_label": cfg.row_label,
                "status": "unavailable",
                "reason": "No matching Model 1B rows found for this row config.",
            })
            continue

        eta_entity = choose_eta_entity(cfg, eta_df)
        if eta_entity is None:
            unavailable_rows.append({
                "row_key": cfg.row_key,
                "row_label": cfg.row_label,
                "status": "unavailable",
                "reason": "No usable elasticity rows found.",
            })
            continue
        eta_sub = pick_eta_rows(eta_df, eta_entity)
        if eta_sub.empty:
            unavailable_rows.append({
                "row_key": cfg.row_key,
                "row_label": cfg.row_label,
                "status": "unavailable",
                "reason": "No usable elasticity rows found after filtering.",
            })
            continue

        lp_sub = lp_sub.copy()
        kl_sub = kl_sub.copy()
        lp_sub["lp_family"] = lp_sub.apply(infer_lp_family, axis=1)
        lp_sub["lp_spec"] = lp_sub.apply(infer_lp_spec, axis=1)
        kl_sub["kl_spec"] = kl_sub.apply(infer_kl_spec, axis=1)

        matched_any = False
        for _, lp_row in lp_sub.iterrows():
            lp_ms = int(lp_row.get("m_start", -999)) if np.isfinite(lp_row.get("m_start", np.nan)) else None
            lp_me = int(lp_row.get("m_end", -999)) if np.isfinite(lp_row.get("m_end", np.nan)) else None
            if lp_ms is None or lp_me is None:
                continue
            if (lp_ms, lp_me) != cfg.preferred_horizon:
                continue

            kl_spec_name = str(lp_row.get("spec_name", "")).lower()
            if "tr" in kl_spec_name:
                kl_spec_name = "ctrl_trend"
            else:
                kl_spec_name = "baseline"
            derived_kl = derive_kl_row(kl_sub, cfg.row_key, kl_spec_name)
            if derived_kl is None:
                continue

            for _, eta_row in eta_sub.iterrows():
                te = float(lp_row.get("beta_hat", np.nan))
                dc = float(derived_kl.get("beta_hat", np.nan))
                eta = float(eta_row.get("eta", np.nan))
                te_se = float(lp_row.get("se", np.nan))
                dc_se = float(derived_kl.get("se", np.nan))
                eta_se = float(eta_row.get("eta_se", np.nan))

                cd, residual, share, sign_consistent, share_valid = compute_accounting(te, dc, eta)
                te_ci_lo, te_ci_hi = ci_from_beta_se(te, te_se)
                dc_ci_lo, dc_ci_hi = ci_from_beta_se(dc, dc_se)
                cd_se, cd_ci_lo, cd_ci_hi = compute_cd_uncertainty(dc, dc_se, eta, eta_se)
                share_se, share_ci_lo, share_ci_hi = compute_share_uncertainty(te, te_se, cd, cd_se)

                eta_pref_flag = int(pd.to_numeric(eta_row.get("preferred_flag", 0), errors="coerce") if pd.notna(eta_row.get("preferred_flag", np.nan)) else 0)
                eta_usage_class = str(eta_row.get("eta_usage_class", ""))

                row = {
                    "row_key": cfg.row_key,
                    "row_label": cfg.row_label,
                    "reform": cfg.reform,
                    "entity": cfg.lp_entity,
                    "horizon": format_horizon(lp_ms, lp_me),
                    "m_start": int(lp_ms),
                    "m_end": int(lp_me),
                    "lp_family": lp_row.get("lp_family", "Unknown"),
                    "lp_spec": lp_row.get("lp_spec", "Baseline"),
                    "lp_design": lp_row.get("design", ""),
                    "lp_spec_name": lp_row.get("spec_name", ""),
                    "lp_window_name": lp_row.get("window_name", ""),
                    "lp_source_file": lp_source_map.get(str(lp_row.get("lp_family", "Unknown")), ""),
                    "lp_target_raw": lp_row.get("target", ""),
                    "kl_spec": "Controls+Trend" if kl_spec_name == "ctrl_trend" else "Baseline",
                    "kl_variant": derived_kl.get("kl_variant", "SummaryDerived"),
                    "kl_design": derived_kl.get("design", ""),
                    "kl_spec_name": derived_kl.get("spec_name", ""),
                    "kl_window_name": derived_kl.get("window_name", ""),
                    "kl_source_file": "intended Model_1B_v8/model1b_binned_window_betas_summary.tsv",
                    "kl_target_raw": cfg.kl_entity,
                    "kl_derivation_note": derived_kl.get("derivation_note", ""),
                    "kl_component_windows": derived_kl.get("component_windows", ""),
                    "kl_component_month_weights": derived_kl.get("component_month_weights", ""),
                    "kl_component_normalized_weights": derived_kl.get("component_normalized_weights", ""),
                    "kl_component_betas": derived_kl.get("component_betas", ""),
                    "kl_component_ses": derived_kl.get("component_ses", ""),
                    "dC_se_method": derived_kl.get("dC_se_method", ""),
                    "eta_entity": eta_entity,
                    "eta_family": eta_row.get("eta_family", ""),
                    "eta_source": eta_row.get("eta_source", ""),
                    "eta_role": eta_row.get("eta_role", ""),
                    "eta_spec": eta_row.get("spec", ""),
                    "eta_preferred_flag": eta_pref_flag,
                    "eta_usage_class": eta_usage_class,
                    "TE": te,
                    "TE_se": te_se,
                    "TE_pvalue": float(lp_row.get("pvalue", np.nan)),
                    "TE_N": float(lp_row.get("N", np.nan)),
                    "TE_R2": float(lp_row.get("R2", np.nan)),
                    "TE_ci_lo": te_ci_lo,
                    "TE_ci_hi": te_ci_hi,
                    "dC": dc,
                    "dC_se": dc_se,
                    "dC_pvalue": float(derived_kl.get("pvalue", np.nan)),
                    "dC_N": float(derived_kl.get("N", np.nan)),
                    "dC_R2": float(derived_kl.get("R2", np.nan)),
                    "dC_ci_lo": dc_ci_lo,
                    "dC_ci_hi": dc_ci_hi,
                    "eta": eta,
                    "eta_se": eta_se,
                    "eta_pvalue": float(eta_row.get("pvalue", np.nan)),
                    "eta_N": float(eta_row.get("N", np.nan)),
                    "eta_R2": float(eta_row.get("R2", np.nan)),
                    "CD": cd,
                    "CD_se": cd_se,
                    "CD_pvalue": two_sided_pvalue(cd, cd_se),
                    "CD_ci_lo": cd_ci_lo,
                    "CD_ci_hi": cd_ci_hi,
                    "residual": residual,
                    "share_explained": share,
                    "share_se": share_se,
                    "share_ci_lo": share_ci_lo,
                    "share_ci_hi": share_ci_hi,
                    "sign_consistent": sign_consistent,
                    "share_valid": share_valid,
                    "eta_uncertainty_included": bool(np.isfinite(eta_se) and eta_se > 0),
                    "manual_aggregate_available": manual_eta_available(eta_df, "Haifa port cluster"),
                    "status": "ok",
                    "reason": "LP exact preferred horizon matched to derived K/L horizon from intended summary bins.",
                }
                row["eta_interpretation_class"] = classify_eta_interpretation(pd.Series(row), cfg, eta_df)
                row["preferred_rule_version"] = "v10_baseline_first_manual_first"
                ok_rows.append(row)
                matched_any = True

        if not matched_any:
            unavailable_rows.append({
                "row_key": cfg.row_key,
                "row_label": cfg.row_label,
                "status": "unavailable",
                "reason": "No usable LP row at preferred horizon and/or no derivable K/L row from intended summary bins.",
            })

    return pd.DataFrame(ok_rows), pd.DataFrame(unavailable_rows)


def build_main_subset(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.Series] = []
    for cfg in ROW_CONFIGS:
        sub = long_df[long_df["row_key"] == cfg.row_key].copy()
        if sub.empty:
            rows.append(pd.Series({
                "row_key": cfg.row_key,
                "row_label": cfg.row_label,
                "horizon": format_horizon(*cfg.preferred_horizon),
                "status": "unavailable",
                "reason": "No supported decomposition rows available.",
            }))
            continue
        best_idx = sorted(sub.index.tolist(), key=lambda idx: preferred_score(sub.loc[idx], cfg))[0]
        row = sub.loc[best_idx].copy()
        row["preferred_rank_tuple"] = str(preferred_score(row, cfg))
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def build_appendix_subset(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return long_df.copy()
    out = long_df.copy()
    order = {cfg.row_key: i for i, cfg in enumerate(ROW_CONFIGS)}
    out["_order"] = out["row_key"].map(order).fillna(99)
    out = out.sort_values([
        "_order", "m_start", "m_end", "lp_family", "lp_spec", "kl_spec", "eta_family", "eta_source"
    ]).drop(columns=["_order"])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    thesis_root = find_thesis_root()

    parser = argparse.ArgumentParser(description="Build the enhanced Model 2 accounting layer.")
    parser.add_argument(
        "--eta",
        type=Path,
        default=thesis_root / "Design" / "Output (new)" / "Model_2_final" / "Tables" / "model2_elasticity_combined.tsv",
    )
    parser.add_argument(
        "--m1a_nyt_window",
        type=Path,
        default=thesis_root / "Design" / "Output (new)" / "Model_1A_v8_2" / "model1a_q_window_betas_nyt.tsv",
    )
    parser.add_argument(
        "--m1a_twfe_window",
        type=Path,
        default=thesis_root / "Design" / "Output (new)" / "Model_1A_v8_2" / "model1a_q_window_betas_twfe.tsv",
    )
    parser.add_argument(
        "--m1b_window",
        type=Path,
        default=thesis_root / "Design" / "Output (new)" / "Model_1B_v8" / "model1b_binned_window_betas_summary.tsv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=thesis_root / "Design" / "Output (new)" / "Model_2_final" / "Tables",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    eta_raw = read_tsv(args.eta)
    if eta_raw is None:
        raise FileNotFoundError(f"Elasticity file not found: {args.eta}")
    eta_df = normalize_eta_df(eta_raw)

    m1a_nyt = normalize_window_df(read_tsv(args.m1a_nyt_window), kind="m1a")
    m1a_twfe = normalize_window_df(read_tsv(args.m1a_twfe_window), kind="m1a")
    m1b_df = normalize_window_df(read_tsv(args.m1b_window), kind="m1b")
    if m1a_nyt is None or m1a_nyt.empty:
        raise FileNotFoundError(f"Model 1A NYT window file missing or empty: {args.m1a_nyt_window}")
    if m1a_twfe is None or m1a_twfe.empty:
        raise FileNotFoundError(f"Model 1A TWFE window file missing or empty: {args.m1a_twfe_window}")
    if m1b_df is None or m1b_df.empty:
        raise FileNotFoundError(f"Model 1B window file missing or empty: {args.m1b_window}")

    m1a_df = pd.concat([m1a_nyt, m1a_twfe], ignore_index=True, sort=False)
    lp_source_map = {
        "NYT": str(args.m1a_nyt_window),
        "Conventional DiD": str(args.m1a_twfe_window),
        "Unknown": str(args.m1a_twfe_window),
    }

    audit_m1a = build_audit(m1a_df, "m1a")
    audit_m1b = build_audit(m1b_df, "m1b")
    coverage_df = build_family_coverage(m1a_df, m1b_df)
    long_df, unavailable_df = build_long_rows(m1a_df, m1b_df, eta_df, lp_source_map)
    main_df = build_main_subset(long_df)
    appendix_df = build_appendix_subset(long_df)
    diagnostics_all = build_diagnostics_table(long_df)
    diagnostics_main = build_diagnostics_table(main_df)

    audit_m1a_path = args.outdir / "model2_accounting_audit_m1a.tsv"
    audit_m1b_path = args.outdir / "model2_accounting_audit_m1b.tsv"
    coverage_path = args.outdir / "model2_accounting_family_coverage.tsv"
    long_path = args.outdir / "model2_accounting_long.tsv"
    main_path = args.outdir / "model2_accounting_main.tsv"
    appendix_path = args.outdir / "model2_accounting_appendix.tsv"
    unavailable_path = args.outdir / "model2_accounting_unavailable.tsv"
    diag_all_path = args.outdir / "model2_diagnostics_all.tsv"
    diag_main_path = args.outdir / "model2_diagnostics_main.tsv"
    pref_rule_path = args.outdir / "model2_preferred_rule.json"
    manifest_path = args.outdir / "model2_accounting_manifest.json"

    audit_m1a.to_csv(audit_m1a_path, sep="\t", index=False)
    audit_m1b.to_csv(audit_m1b_path, sep="\t", index=False)
    coverage_df.to_csv(coverage_path, sep="\t", index=False)
    long_df.to_csv(long_path, sep="\t", index=False)
    main_df.to_csv(main_path, sep="\t", index=False)
    appendix_df.to_csv(appendix_path, sep="\t", index=False)
    unavailable_df.to_csv(unavailable_path, sep="\t", index=False)
    diagnostics_all.to_csv(diag_all_path, sep="\t", index=False)
    diagnostics_main.to_csv(diag_main_path, sep="\t", index=False)

    preference_rule = {
        "version": "v10_baseline_first_manual_first",
        "logic": [
            "preferred horizon exact match",
            "LP family order by row config",
            "Baseline before +Tr on LP side",
            "Baseline before Controls+Trend on KL side",
            "preferred-manual before aggregate-regression fallback before other robustness rows",
            "sign-consistent rows before inconsistent rows",
            "share-valid rows before mechanically problematic rows",
            "smaller TE_se then smaller dC_se then higher TE_R2 as tie-breakers",
        ],
    }
    pref_rule_path.write_text(json.dumps(preference_rule, indent=2), encoding="utf-8")

    manifest = {
        "script": "Model_2_step3_accounting_v12.py",
        "inputs": {
            "eta": str(args.eta),
            "model1a_nyt_window": str(args.m1a_nyt_window),
            "model1a_twfe_window": str(args.m1a_twfe_window),
            "model1b_window": str(args.m1b_window),
        },
        "outputs": {
            "audit_m1a": str(audit_m1a_path),
            "audit_m1b": str(audit_m1b_path),
            "family_coverage": str(coverage_path),
            "long": str(long_path),
            "main": str(main_path),
            "appendix": str(appendix_path),
            "unavailable": str(unavailable_path),
            "diagnostics_all": str(diag_all_path),
            "diagnostics_main": str(diag_main_path),
            "preferred_rule": str(pref_rule_path),
        },
        "rows_long": int(len(long_df)),
        "rows_main": int(len(main_df)),
        "rows_appendix": int(len(appendix_df)),
        "rows_unavailable": int(len(unavailable_df)),
        "rows_family_coverage": int(len(coverage_df)),
        "note": "v11 keeps the v10 accounting logic but recognizes the new preferred manual aggregate proxy when available.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== Model_2_step3_accounting_v12.py: done ===")
    print(f"Elasticity input : {args.eta}")
    print(f"Model 1A NYT     : {args.m1a_nyt_window}")
    print(f"Model 1A TWFE    : {args.m1a_twfe_window}")
    print(f"Model 1B windows : {args.m1b_window}")
    print(f"Audit M1A        : {audit_m1a_path} (rows={len(audit_m1a)})")
    print(f"Audit M1B        : {audit_m1b_path} (rows={len(audit_m1b)})")
    print(f"Coverage out     : {coverage_path} (rows={len(coverage_df)})")
    print(f"Long out         : {long_path} (rows={len(long_df)})")
    print(f"Main out         : {main_path} (rows={len(main_df)})")
    print(f"Appendix out     : {appendix_path} (rows={len(appendix_df)})")
    print(f"Unavailable out  : {unavailable_path} (rows={len(unavailable_df)})")
    print(f"Diag all out     : {diag_all_path} (rows={len(diagnostics_all)})")
    print(f"Diag main out    : {diag_main_path} (rows={len(diagnostics_main)})")
    print(f"Preference rule  : {pref_rule_path}")
    print(f"Manifest         : {manifest_path}")

    if len(long_df) == 0:
        raise ValueError(
            "Model 2 accounting produced zero usable decomposition rows even after intended-file matching and K/L horizon derivation."
        )


if __name__ == "__main__":
    main()
