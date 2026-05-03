#!/usr/bin/env python3
"""
Model_2_accounting(v3)_final.py

Build the corrected Model 2 accounting decomposition.

Main fixes relative to the first v3 accounting file
--------------------------------------------------
1. Correctly recognizes Model 1A / Model 1B window columns named `a` and `b`.
2. Never silently falls back to the first matching row when the requested window is not found.
3. Requires exact post-window matches for causal decomposition rows.
4. Keeps unavailable rows explicitly, but prevents unavailable decomposition cells from being
   accidentally interpreted as valid accounting results.
5. Adds sign/validity diagnostics for the share-explained calculation.

Inputs
------
- Design/Output (new)/Model_2/Tables/model2_elasticity_combined.tsv
- Design/Output (new)/Model_1A/model1a_lp_window_betas_all.tsv
- Design/Output (new)/Model_1B_relaxed/model1b_kl_window_betas_all_relaxed.tsv

Outputs
-------
Design/Output (new)/Model_2/Tables/
  - model2_accounting_long.tsv
  - model2_accounting_preferred.tsv
  - model2_accounting_elasticity_robustness.tsv
  - model2_accounting_unavailable_rows.tsv
  - model2_accounting_manifest.json
"""

from __future__ import annotations

import argparse
import json
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
    raise FileNotFoundError("Could not locate thesis root. Run this from inside the Thesis project.")


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


def normalize_window_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Normalize Model 1A / Model 1B window-estimate files.

    This is intentionally strict about windows. The old bug came from not recognizing
    columns called `a` and `b`, which caused the lookup to grab the first matching row,
    often avg_pre. This function now maps `a -> m_start` and `b -> m_end`.
    """
    if df is None:
        return None

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    ren = {}
    for cand_list, dst in [
        (["reform", "reform_key"], "reform"),
        (["target", "entity", "unit", "series_id"], "target"),
        (["design", "estimator_family", "model_type"], "design"),
        (["spec_name", "fe_type", "spec", "spec_key", "specification"], "spec_name"),
        (["window_name", "window", "horizon_key", "horizon", "window_label"], "window_name"),
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
    missing_required = [c for c in required if c not in out.columns]
    if missing_required:
        raise ValueError(f"Window file is missing required columns after normalization: {missing_required}")

    for c in ["reform", "target", "design", "spec_name", "window_name"]:
        if c in out.columns:
            out[c] = normalize_string_col(out[c])

    for c in ["m_start", "m_end", "beta_hat", "se", "pvalue", "N", "R2"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def ensure_window_columns(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Make window-column handling robust even if an upstream file uses unexpected aliases.

    The earlier crash happened because a downstream lookup still encountered a file whose
    window columns were not named m_start / m_end at runtime. This helper re-applies the
    alias logic defensively and guarantees that lookup_window_beta never crashes with a
    KeyError when the schema is merely non-standard.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    # Re-run the standard normalization first.
    out = normalize_window_df(out)
    if out is None or out.empty:
        return out

    fallback_aliases = [
        (["m_start", "start_m", "window_start", "event_start", "a"], "m_start"),
        (["m_end", "end_m", "window_end", "event_end", "b"], "m_end"),
        (["beta_hat", "beta", "coef", "coefficient", "estimate", "avg_beta"], "beta_hat"),
        (["N", "n_obs", "n", "obs", "observations"], "N"),
        (["R2", "r2", "within_r2", "rsquared", "r_squared"], "R2"),
    ]
    ren = {}
    for cand_list, dst in fallback_aliases:
        if dst in out.columns:
            continue
        c = first_present(out.columns, cand_list)
        if c is not None:
            ren[c] = dst
    if ren:
        out = out.rename(columns=ren)
    out = collapse_duplicate_columns(out)

    for c in ["m_start", "m_end", "beta_hat", "se", "pvalue", "N", "R2"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate columns by taking first non-null row-wise."""
    if df is None or df.empty or not df.columns.duplicated().any():
        return df
    out = pd.DataFrame(index=df.index)
    seen=[]
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


def get_named_series(df: pd.DataFrame, name: str):
    if df is None or name not in df.columns:
        return None
    cols = df.loc[:, df.columns == name]
    s = cols.iloc[:, 0].copy()
    for j in range(1, cols.shape[1]):
        s = s.combine_first(cols.iloc[:, j])
    return s


def normalize_eta_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    ren = {}
    for cand_list, dst in [
        (["entity"], "entity"),
        (["eta_family", "family"], "eta_family"),
        (["eta_source", "source"], "eta_source"),
        (["eta_role", "role"], "eta_role"),
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

    if "status" not in out.columns:
        out["status"] = "ok"
    if "preferred_flag" not in out.columns:
        out["preferred_flag"] = 0
    if "eta_family" not in out.columns:
        out["eta_family"] = ""
    if "eta_role" not in out.columns:
        out["eta_role"] = ""
    if "spec" not in out.columns:
        out["spec"] = ""
    if "reason" not in out.columns:
        out["reason"] = ""

    for c in ["entity", "eta_family", "eta_source", "eta_role", "spec", "status", "reason"]:
        if c in out.columns:
            out[c] = normalize_string_col(out[c])

    for c in ["eta", "eta_se", "preferred_flag", "delta"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ---------------------------------------------------------------------
# Aliases and row config
# ---------------------------------------------------------------------

REFORM_ALIASES = {
    "haifa_comp": ["haifa_comp", "haifa competition", "competition_haifa", "competition"],
    "haifa_priv": ["haifa_priv", "haifa privatization", "privatization_haifa", "privatization"],
}

TARGET_ALIASES_TE = {
    "Haifa--Legacy": [
        "Haifa--Legacy", "Haifa-Legacy", "Haifa Legacy", "Haifa_Legacy",
        "Haifa-Legacy terminal", "Haifa-Legacy terminal LP", "Haifa_Legacy_Q",
    ],
    "Haifa port cluster": [
        "Haifa port cluster", "Haifa port", "Haifa", "Haifa aggregate", "Haifa_port_Q", "Haifa_port",
    ],
}

TARGET_ALIASES_DC = {
    "Haifa--Legacy": [
        "Haifa--Legacy", "Haifa-Legacy", "Haifa Legacy", "Haifa_Legacy",
        "Haifa-Legacy K/L", "Haifa_Legacy_KL", "Haifa_Legacy_KL_relaxed",
    ],
    "Haifa--Bayport": [
        "Haifa--Bayport", "Haifa-Bayport", "Haifa Bayport", "Haifa_Bayport", "Haifa_Bayport_KL",
    ],
    "Haifa port cluster": [
        "Haifa port cluster", "Haifa port", "Haifa", "Haifa_port_KL", "Haifa port K/L", "Haifa_port",
    ],
}


def clean_token(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower().replace("_", " ").replace("--", "-").replace("  ", " ")


def any_match(value: object, aliases: Sequence[str]) -> bool:
    v = clean_token(value)
    return v in {clean_token(a) for a in aliases}


def spec_matches(value: object, preferred_specs: Sequence[str]) -> bool:
    v = clean_token(value)
    return any(v == clean_token(s) for s in preferred_specs)


@dataclass(frozen=True)
class RowSpec:
    row_key: str
    row_label: str
    reform: str
    entity: str
    estimator_family: str
    horizon_label: str
    m_start: int
    m_end: int
    preferred_eta_entity: str
    preferred_eta_source: str
    te_required: bool
    dc_required: bool
    availability_reason: Optional[str] = None


ROW_SPECS: List[RowSpec] = [
    RowSpec(
        row_key="competition_legacy",
        row_label="Competition - Legacy",
        reform="haifa_comp",
        entity="Haifa--Legacy",
        estimator_family="NYT",
        horizon_label="[1,13]",
        m_start=1,
        m_end=13,
        preferred_eta_entity="Haifa--Legacy",
        preferred_eta_source="HPC_labor_share",
        te_required=False,
        dc_required=False,
        availability_reason="competition K/L decomposition not estimated under current Haifa-only capital design",
    ),
    RowSpec(
        row_key="competition_aggregate",
        row_label="Competition - Aggregate",
        reform="haifa_comp",
        entity="Haifa port cluster",
        estimator_family="NYT",
        horizon_label="[1,13]",
        m_start=1,
        m_end=13,
        preferred_eta_entity="Haifa--Legacy",
        preferred_eta_source="HPC_labor_share",
        te_required=False,
        dc_required=False,
        availability_reason="aggregate competition K/L decomposition not estimated under current Haifa-only capital design",
    ),
    RowSpec(
        row_key="privatization_legacy",
        row_label="Privatization - Legacy",
        reform="haifa_priv",
        entity="Haifa--Legacy",
        estimator_family="NYT",
        horizon_label="[1,8]",
        m_start=1,
        m_end=8,
        preferred_eta_entity="Haifa--Legacy",
        preferred_eta_source="HPC_labor_share",
        te_required=True,
        dc_required=True,
        availability_reason=None,
    ),
    RowSpec(
        row_key="privatization_aggregate",
        row_label="Privatization - Aggregate",
        reform="haifa_priv",
        entity="Haifa port cluster",
        estimator_family="NYT",
        horizon_label="[1,8]",
        m_start=1,
        m_end=8,
        preferred_eta_entity="Haifa--Legacy",
        preferred_eta_source="HPC_labor_share",
        te_required=False,
        dc_required=False,
        availability_reason="aggregate privatization K/L comparison unavailable under current design",
    ),
]


# ---------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------

def lookup_window_beta(
    df: Optional[pd.DataFrame],
    reform: str,
    target: str,
    m_start: int,
    m_end: int,
    preferred_specs: Sequence[str],
    target_aliases: Dict[str, Sequence[str]],
) -> Tuple[float, float, float, float, str, str, str]:
    """Look up an exact window beta.

    Returns: beta, se, N, R2, spec_name, status, reason.

    This function deliberately refuses to use a first-row fallback if exact window
    columns are absent or the exact requested window is missing.
    """
    df = ensure_window_columns(df)
    if df is None or df.empty:
        return (np.nan, np.nan, np.nan, np.nan, "", "missing_input", "input file missing or empty")

    d = collapse_duplicate_columns(df.copy())
    reform_s = get_named_series(d, "reform")
    target_s = get_named_series(d, "target")
    mstart_s = get_named_series(d, "m_start")
    mend_s = get_named_series(d, "m_end")
    beta_s = get_named_series(d, "beta_hat")
    missing = [name for name, s in [("reform", reform_s), ("target", target_s), ("m_start", mstart_s), ("m_end", mend_s), ("beta_hat", beta_s)] if s is None]
    if missing:
        return (
            np.nan, np.nan, np.nan, np.nan, "", "bad_schema",
            f"window file missing required normalized columns: {missing}",
        )

    aliases = target_aliases.get(target, [target])
    mask = reform_s.apply(lambda x: any_match(x, REFORM_ALIASES[reform]))
    mask = mask & target_s.apply(lambda x: any_match(x, aliases))
    mask = mask & (pd.to_numeric(mstart_s, errors="coerce") == m_start)
    mask = mask & (pd.to_numeric(mend_s, errors="coerce") == m_end)
    d = d.loc[mask].copy()

    if d.empty:
        return (
            np.nan, np.nan, np.nan, np.nan, "", "unavailable",
            f"exact requested post window [{m_start},{m_end}] not found",
        )

    spec_s = get_named_series(d, "spec_name")
    if spec_s is not None and spec_s.notna().any():
        for spec in preferred_specs:
            dd = d.loc[spec_s.apply(lambda x: spec_matches(x, [spec]))].copy()
            if not dd.empty:
                r = dd.iloc[0]
                return (
                    float(r.get("beta_hat", np.nan)),
                    float(r.get("se", np.nan)),
                    float(r.get("N", np.nan)),
                    float(r.get("R2", np.nan)),
                    str(r.get("spec_name", spec)),
                    "ok",
                    "exact requested window and preferred spec found",
                )

        spec_s = get_named_series(d, "spec_name")
        available_specs = [] if spec_s is None else sorted(spec_s.dropna().astype(str).unique().tolist())
        return (
            np.nan, np.nan, np.nan, np.nan, "", "unavailable",
            f"exact window found, but none of preferred specs {list(preferred_specs)} found; available specs={available_specs}",
        )

    # Exact window exists, but there is no spec column. Use it because the exact window is valid.
    r = d.iloc[0]
    return (
        float(r.get("beta_hat", np.nan)),
        float(r.get("se", np.nan)),
        float(r.get("N", np.nan)),
        float(r.get("R2", np.nan)),
        "",
        "ok",
        "exact requested window found; no spec column supplied",
    )


def pick_preferred_eta(eta_df: pd.DataFrame, entity: str, source: str) -> Tuple[float, float, str, str]:
    sub = eta_df[
        (eta_df["entity"].astype(str) == entity)
        & (eta_df["eta_source"].astype(str) == source)
        & (eta_df["status"].fillna("ok").astype(str).str.lower() != "unavailable")
    ].copy()
    if sub.empty:
        return (np.nan, np.nan, "unavailable", "preferred elasticity row not found")
    if len(sub) > 1:
        # Prefer the row explicitly flagged as preferred if present.
        flagged = sub[pd.to_numeric(sub.get("preferred_flag", 0), errors="coerce").fillna(0) == 1]
        if len(flagged) == 1:
            sub = flagged
        else:
            return (np.nan, np.nan, "bad_schema", "multiple candidate preferred elasticity rows found")
    r = sub.iloc[0]
    return (float(r.get("eta", np.nan)), float(r.get("eta_se", np.nan)), "ok", "preferred elasticity found")


def compute_accounting(TE: float, dC: float, eta: float, status: str) -> Tuple[float, float, float, bool, bool]:
    if status != "ok" or not (np.isfinite(TE) and np.isfinite(dC) and np.isfinite(eta)):
        return (np.nan, np.nan, np.nan, False, False)
    CD = eta * dC
    residual = TE - CD
    share = CD / TE if TE != 0 else np.nan
    sign_consistent = bool(np.sign(TE) == np.sign(CD)) if TE != 0 and CD != 0 else False
    share_valid = bool(np.isfinite(share) and sign_consistent and abs(TE) > 1e-10)
    return (CD, residual, share, sign_consistent, share_valid)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    thesis_root = find_thesis_root()

    parser = argparse.ArgumentParser(description="Build corrected Model 2 accounting outputs.")
    parser.add_argument(
        "--eta",
        type=Path,
        default=thesis_root / "Design" / "Output (new)" / "Model_2" / "Tables" / "model2_elasticity_combined.tsv",
    )
    parser.add_argument(
        "--m1a_window",
        type=Path,
        default=thesis_root / "Design" / "Output (new)" / "Model_1A" / "model1a_lp_window_betas_all.tsv",
    )
    parser.add_argument(
        "--m1b_window_relaxed",
        type=Path,
        default=thesis_root / "Design" / "Output (new)" / "Model_1B_relaxed" / "model1b_kl_window_betas_all_relaxed.tsv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=thesis_root / "Design" / "Output (new)" / "Model_2" / "Tables",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    eta_raw = read_tsv(args.eta)
    if eta_raw is None:
        raise FileNotFoundError(f"Elasticity file not found: {args.eta}")
    eta_df = normalize_eta_df(eta_raw)

    m1a_w = ensure_window_columns(read_tsv(args.m1a_window))
    m1b_w = ensure_window_columns(read_tsv(args.m1b_window_relaxed))

    if m1a_w is not None:
        print(f"Normalized Model 1A window columns: {list(m1a_w.columns)}")
    if m1b_w is not None:
        print(f"Normalized Model 1B window columns: {list(m1b_w.columns)}")

    rows: List[Dict[str, object]] = []

    for spec in ROW_SPECS:
        te, te_se, te_n, te_r2, te_spec, te_status, te_reason = lookup_window_beta(
            m1a_w,
            reform=spec.reform,
            target=spec.entity,
            m_start=spec.m_start,
            m_end=spec.m_end,
            preferred_specs=["porttr", "relaxed_tr", "baseline"],
            target_aliases=TARGET_ALIASES_TE,
        )

        dc, dc_se, dc_n, dc_r2, dc_spec, dc_status, dc_reason = lookup_window_beta(
            m1b_w,
            reform=spec.reform,
            target=spec.entity,
            m_start=spec.m_start,
            m_end=spec.m_end,
            preferred_specs=["relaxed_tr", "relaxed", "ts_trend", "porttr", "baseline"],
            target_aliases=TARGET_ALIASES_DC,
        )

        eta, eta_se, eta_status, eta_reason = pick_preferred_eta(
            eta_df, spec.preferred_eta_entity, spec.preferred_eta_source
        )

        status = "ok"
        reason = "exact supported TE and dC windows found"

        if spec.availability_reason is not None:
            status = "unavailable"
            reason = spec.availability_reason
        else:
            if spec.te_required and te_status != "ok":
                status = "unavailable"
                reason = f"TE unavailable: {te_reason}"
            elif spec.te_required and not np.isfinite(te):
                status = "unavailable"
                reason = "TE unavailable: estimate is not finite"
            elif spec.dc_required and dc_status != "ok":
                status = "unavailable"
                reason = f"dC unavailable: {dc_reason}"
            elif spec.dc_required and not np.isfinite(dc):
                status = "unavailable"
                reason = "dC unavailable: estimate is not finite"
            elif eta_status != "ok" or not np.isfinite(eta):
                status = "unavailable"
                reason = f"eta unavailable: {eta_reason}"

        CD, residual, share, sign_consistent, share_valid = compute_accounting(TE=te, dC=dc, eta=eta, status=status)

        rows.append({
            "row_key": spec.row_key,
            "row_label": spec.row_label,
            "reform": spec.reform,
            "entity": spec.entity,
            "estimator_family": spec.estimator_family,
            "horizon": spec.horizon_label,
            "m_start": spec.m_start,
            "m_end": spec.m_end,
            "TE": te,
            "TE_se": te_se,
            "TE_N": te_n,
            "TE_R2": te_r2,
            "TE_status": te_status,
            "TE_reason": te_reason,
            "dC": dc,
            "dC_se": dc_se,
            "dC_N": dc_n,
            "dC_R2": dc_r2,
            "dC_status": dc_status,
            "dC_reason": dc_reason,
            "eta": eta,
            "eta_se": eta_se,
            "eta_source": spec.preferred_eta_source,
            "eta_entity": spec.preferred_eta_entity,
            "eta_status": eta_status,
            "eta_reason": eta_reason,
            "CD": CD,
            "residual": residual,
            "share_explained": share,
            "sign_consistent": sign_consistent,
            "share_valid": share_valid,
            "spec_te": te_spec,
            "spec_dC": dc_spec,
            "status": status,
            "reason": reason,
        })

    long_df = pd.DataFrame(rows)

    # Elasticity robustness for the one currently feasible causal decomposition row.
    rob_rows: List[Dict[str, object]] = []
    target_row = long_df[long_df["row_key"] == "privatization_legacy"].copy()
    if not target_row.empty:
        base = target_row.iloc[0].to_dict()
        eta_sub = eta_df[
            (eta_df["entity"].astype(str) == "Haifa--Legacy")
            & (eta_df["status"].fillna("ok").astype(str).str.lower() != "unavailable")
            & (pd.to_numeric(eta_df["eta"], errors="coerce").notna())
        ].copy()
        for _, r in eta_sub.iterrows():
            row = base.copy()
            row["eta"] = float(r.get("eta", np.nan))
            row["eta_se"] = float(r.get("eta_se", np.nan)) if pd.notna(r.get("eta_se", np.nan)) else np.nan
            row["eta_source"] = r.get("eta_source", "")
            row["eta_family"] = r.get("eta_family", "")
            row["eta_role"] = r.get("eta_role", "")
            row["eta_spec"] = r.get("spec", "")

            if base["status"] == "ok" and np.isfinite(row["dC"]) and np.isfinite(row["eta"]) and np.isfinite(row["TE"]):
                CD, residual, share, sign_consistent, share_valid = compute_accounting(
                    TE=float(row["TE"]), dC=float(row["dC"]), eta=float(row["eta"]), status="ok"
                )
                row["CD"] = CD
                row["residual"] = residual
                row["share_explained"] = share
                row["sign_consistent"] = sign_consistent
                row["share_valid"] = share_valid
                row["status"] = "ok"
                row["reason"] = "elasticity robustness row"
            else:
                row["CD"] = np.nan
                row["residual"] = np.nan
                row["share_explained"] = np.nan
                row["sign_consistent"] = False
                row["share_valid"] = False
                row["status"] = "unavailable"
                row["reason"] = "elasticity robustness row unavailable because preferred TE or dC is missing"
            rob_rows.append(row)
    rob_df = pd.DataFrame(rob_rows)

    preferred_df = long_df[(long_df["row_key"] == "privatization_legacy") | (long_df["status"] == "unavailable")].copy()
    unavailable_df = long_df[long_df["status"] != "ok"].copy()

    long_path = args.outdir / "model2_accounting_long.tsv"
    pref_path = args.outdir / "model2_accounting_preferred.tsv"
    rob_path = args.outdir / "model2_accounting_elasticity_robustness.tsv"
    unavail_path = args.outdir / "model2_accounting_unavailable_rows.tsv"
    manifest_path = args.outdir / "model2_accounting_manifest.json"

    long_df.to_csv(long_path, sep="\t", index=False)
    preferred_df.to_csv(pref_path, sep="\t", index=False)
    rob_df.to_csv(rob_path, sep="\t", index=False)
    unavailable_df.to_csv(unavail_path, sep="\t", index=False)

    manifest = {
        "script": "Model_2_accounting(v3)_final2.py",
        "inputs": {
            "eta": str(args.eta),
            "model1a_window": str(args.m1a_window),
            "model1b_window_relaxed": str(args.m1b_window_relaxed),
        },
        "outputs": {
            "long": str(long_path),
            "preferred": str(pref_path),
            "elasticity_robustness": str(rob_path),
            "unavailable": str(unavail_path),
        },
        "rows_total": int(len(long_df)),
        "rows_ok": int((long_df["status"] == "ok").sum()),
        "rows_unavailable": int((long_df["status"] != "ok").sum()),
        "important_fix": "Window columns named a/b are normalized to m_start/m_end, and exact-window lookup is required.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== Model_2_accounting(v3)_final2: done ===")
    print(f"Elasticity input  : {args.eta}")
    print(f"Model 1A windows  : {args.m1a_window}")
    print(f"Model 1B windows  : {args.m1b_window_relaxed}")
    print(f"Long out          : {long_path} (rows={len(long_df)}, ok={(long_df['status'] == 'ok').sum()})")
    print(f"Preferred out     : {pref_path} (rows={len(preferred_df)})")
    print(f"Robustness out    : {rob_path} (rows={len(rob_df)})")
    print(f"Unavailable out   : {unavail_path} (rows={len(unavailable_df)})")
    print(f"Manifest          : {manifest_path}")

    # Print the key row in the console so a bad window selection is easy to catch.
    key = long_df[long_df["row_key"] == "privatization_legacy"]
    if not key.empty:
        r = key.iloc[0]
        print("\nKey preferred row: privatization_legacy")
        print(f"  status          : {r['status']}")
        print(f"  TE              : {r['TE']}")
        print(f"  dC              : {r['dC']}")
        print(f"  eta             : {r['eta']}")
        print(f"  CD              : {r['CD']}")
        print(f"  share_explained : {r['share_explained']}")
        print(f"  spec_te/spec_dC : {r['spec_te']} / {r['spec_dC']}")
        print(f"  reason          : {r['reason']}")


if __name__ == "__main__":
    main()
