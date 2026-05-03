#!/usr/bin/env python3
"""
Model_2_elasticity(v3).py

Build the new Model 2 elasticity layer.

This script produces two families of elasticity values:
1) manual / calibrated elasticities (preferred)
2) regression-based elasticities (robustness / comparison)

Outputs
-------
Design/Output (new)/Model_2/Tables/
  - model2_elasticity_manual.tsv
  - model2_elasticity_regression.tsv
  - model2_elasticity_combined.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


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
# Regression helpers
# ---------------------------------------------------------------------

def run_ols(df: pd.DataFrame, formula: str):
    model = smf.ols(formula=formula, data=df)
    return model.fit(cov_type="HC1")


def extract_logkl(res):
    return {
        "eta": float(res.params.get("log_KL", np.nan)),
        "eta_se": float(res.bse.get("log_KL", np.nan)),
        "pvalue": float(res.pvalues.get("log_KL", np.nan)),
        "N": int(round(res.nobs)),
        "R2": float(res.rsquared),
    }


# ---------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------

def build_manual_rows() -> pd.DataFrame:
    rows = [
        {
            "entity": "Haifa--Legacy",
            "eta_family": "manual",
            "eta_source": "HPC_labor_share",
            "eta_role": "preferred",
            "spec": "manual",
            "delta": np.nan,
            "eta": 0.251,
            "eta_se": np.nan,
            "pvalue": np.nan,
            "N": np.nan,
            "R2": np.nan,
            "preferred_flag": 1,
            "status": "ok",
            "reason": "preferred manual labor-share / Cobb-Douglas elasticity",
            "notes": "Entity-specific terminal-operator labor-share alpha for Haifa-Legacy.",
        },
        {
            "entity": "Haifa--Legacy",
            "eta_family": "manual",
            "eta_source": "one_third_benchmark",
            "eta_role": "robustness",
            "spec": "manual",
            "delta": np.nan,
            "eta": 1.0 / 3.0,
            "eta_se": np.nan,
            "pvalue": np.nan,
            "N": np.nan,
            "R2": np.nan,
            "preferred_flag": 0,
            "status": "ok",
            "reason": "standard Cobb-Douglas benchmark",
            "notes": "Common alpha=1/3 benchmark used for robustness.",
        },
        {
            "entity": "Haifa--Legacy",
            "eta_family": "manual",
            "eta_source": "hazan_tsur_benchmark",
            "eta_role": "robustness",
            "spec": "manual",
            "delta": np.nan,
            "eta": 0.420,
            "eta_se": np.nan,
            "pvalue": np.nan,
            "N": np.nan,
            "R2": np.nan,
            "preferred_flag": 0,
            "status": "ok",
            "reason": "literature benchmark",
            "notes": "Israeli benchmark from related productivity-accounting discussion.",
        },
        {
            "entity": "Haifa--Bayport",
            "eta_family": "manual",
            "eta_source": "SIPG_group_proxy",
            "eta_role": "diagnostic",
            "spec": "manual",
            "delta": np.nan,
            "eta": 0.658,
            "eta_se": np.nan,
            "pvalue": np.nan,
            "N": np.nan,
            "R2": np.nan,
            "preferred_flag": 0,
            "status": "ok",
            "reason": "diagnostic only",
            "notes": "Group-level SIPG labor-share proxy, not preferred for main accounting.",
        },
        {
            "entity": "Haifa port cluster",
            "eta_family": "manual",
            "eta_source": "none",
            "eta_role": "diagnostic",
            "spec": "manual",
            "delta": np.nan,
            "eta": np.nan,
            "eta_se": np.nan,
            "pvalue": np.nan,
            "N": np.nan,
            "R2": np.nan,
            "preferred_flag": 0,
            "status": "unavailable",
            "reason": "no preferred manual aggregate elasticity currently specified",
            "notes": "Aggregate/cluster manual elasticity intentionally left unspecified.",
        },
    ]
    return pd.DataFrame(rows)



def normalize_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    # enforce core types
    for c in ["log_LP", "log_KL", "t_index", "month_index", "quarter_index", "delta"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "t_index" not in df.columns:
        if "quarter_index" in df.columns:
            order = pd.Series(df["quarter_index"]).rank(method="dense")
            df["t_index"] = pd.to_numeric(order, errors="coerce")
        elif "month_index" in df.columns:
            # dense rank so gaps in month_index are ok
            order = pd.Series(df["month_index"]).rank(method="dense")
            df["t_index"] = pd.to_numeric(order, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df



def estimate_regressions(term: pd.DataFrame, cluster: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    # Determine delta groups if present, otherwise one pooled run
    delta_values = []
    if "delta" in term.columns and term["delta"].notna().any():
        delta_values = sorted(term["delta"].dropna().round(6).unique().tolist())
    else:
        delta_values = [np.nan]

    for delta in delta_values:
        if pd.isna(delta):
            term_d = term.copy()
            cluster_d = cluster.copy()
        else:
            term_d = term[np.isclose(term["delta"], delta, equal_nan=False)].copy()
            cluster_d = cluster[np.isclose(cluster["delta"], delta, equal_nan=False)].copy() if "delta" in cluster.columns else cluster.copy()

        # single-entity time-series specs
        for entity, source in [("Haifa--Legacy", "legacy_ts_trend"), ("Haifa--Bayport", "bayport_ts_trend")]:
            sub = term_d[term_d["entity"] == entity].copy()
            if len(sub.dropna(subset=["log_LP", "log_KL", "t_index"])) < 8:
                rows.append({
                    "entity": entity,
                    "eta_family": "regression",
                    "eta_source": source,
                    "eta_role": "robustness",
                    "spec": "TS+trend",
                    "delta": delta,
                    "eta": np.nan,
                    "eta_se": np.nan,
                    "pvalue": np.nan,
                    "N": len(sub),
                    "R2": np.nan,
                    "preferred_flag": 0,
                    "status": "unavailable",
                    "reason": "insufficient usable rows for regression",
                    "notes": "Requires at least 8 non-missing observations.",
                })
                continue
            sub = sub.dropna(subset=["log_LP", "log_KL", "t_index"])
            res = run_ols(sub, "log_LP ~ log_KL + t_index")
            vals = extract_logkl(res)
            rows.append({
                "entity": entity,
                "eta_family": "regression",
                "eta_source": source,
                "eta_role": "robustness",
                "spec": "TS+trend",
                "delta": delta,
                **vals,
                "preferred_flag": 0,
                "status": "ok",
                "reason": "HC1 time-series regression",
                "notes": "Reduced-form co-movement estimate, not structural.",
            })

        # pooled terminal regression
        pooled = term_d.dropna(subset=["log_LP", "log_KL", "t_index", "entity"]).copy()
        if len(pooled) >= 12 and pooled["entity"].nunique() >= 2:
            res = run_ols(pooled, "log_LP ~ log_KL + t_index + C(entity)")
            vals = extract_logkl(res)
            rows.append({
                "entity": "Pooled Haifa terminals",
                "eta_family": "regression",
                "eta_source": "pooled_trend_fe",
                "eta_role": "robustness",
                "spec": "Pooled+trend+FE",
                "delta": delta,
                **vals,
                "preferred_flag": 0,
                "status": "ok",
                "reason": "HC1 pooled regression",
                "notes": "Pooled terminal regression with entity fixed effects and linear time trend.",
            })
        else:
            rows.append({
                "entity": "Pooled Haifa terminals",
                "eta_family": "regression",
                "eta_source": "pooled_trend_fe",
                "eta_role": "robustness",
                "spec": "Pooled+trend+FE",
                "delta": delta,
                "eta": np.nan,
                "eta_se": np.nan,
                "pvalue": np.nan,
                "N": len(pooled),
                "R2": np.nan,
                "preferred_flag": 0,
                "status": "unavailable",
                "reason": "insufficient pooled overlap",
                "notes": "Requires both terminal entities and at least 12 usable rows.",
            })

        # optional cluster regression
        cl = cluster_d.dropna(subset=["log_LP", "log_KL", "t_index"]).copy() if not cluster_d.empty else cluster_d.copy()
        if len(cl) >= 8:
            res = run_ols(cl, "log_LP ~ log_KL + t_index")
            vals = extract_logkl(res)
            rows.append({
                "entity": "Haifa port cluster",
                "eta_family": "regression",
                "eta_source": "cluster_ts_trend",
                "eta_role": "robustness",
                "spec": "TS+trend",
                "delta": delta,
                **vals,
                "preferred_flag": 0,
                "status": "ok",
                "reason": "HC1 aggregate time-series regression",
                "notes": "Aggregate-port robustness regression only.",
            })
        else:
            rows.append({
                "entity": "Haifa port cluster",
                "eta_family": "regression",
                "eta_source": "cluster_ts_trend",
                "eta_role": "robustness",
                "spec": "TS+trend",
                "delta": delta,
                "eta": np.nan,
                "eta_se": np.nan,
                "pvalue": np.nan,
                "N": len(cl),
                "R2": np.nan,
                "preferred_flag": 0,
                "status": "unavailable",
                "reason": "insufficient cluster overlap",
                "notes": "Aggregate-port elasticity is optional and only estimated when overlap exists.",
            })

    return pd.DataFrame(rows)



def main() -> None:
    thesis_root = find_thesis_root()

    parser = argparse.ArgumentParser(description="Build the new Model 2 elasticity layer.")
    preferred_terminal = thesis_root / "Design" / "Output (new)" / "Model_2" / "Inputs" / "model2_terminal_panel_quarterly.tsv"
    fallback_terminal = thesis_root / "Design" / "Output (new)" / "Model_2" / "Inputs" / "model2_terminal_panel.tsv"

    parser.add_argument(
        "--terminal_panel",
        type=Path,
        default=preferred_terminal if preferred_terminal.exists() else fallback_terminal,
    )
    parser.add_argument(
        "--cluster_panel",
        type=Path,
        default=thesis_root / "Design" / "Output (new)" / "Model_2" / "Inputs" / "model2_cluster_panel.tsv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=thesis_root / "Design" / "Output (new)" / "Model_2" / "Tables",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.terminal_panel.exists():
        raise FileNotFoundError(f"Terminal panel not found: {args.terminal_panel}")
    quarterly_sibling = args.terminal_panel.with_name("model2_terminal_panel_quarterly.tsv")
    if args.terminal_panel.name == "model2_terminal_panel.tsv" and quarterly_sibling.exists():
        print("[INFO] Detected monthly terminal overlap panel, but quarterly terminal panel exists.")
        print(f"[INFO] Using quarterly terminal panel for elasticity estimation: {quarterly_sibling}")
        args.terminal_panel = quarterly_sibling
    terminal = normalize_panel(args.terminal_panel)
    cluster = normalize_panel(args.cluster_panel) if args.cluster_panel.exists() else pd.DataFrame(columns=terminal.columns)

    manual = build_manual_rows()
    regression = estimate_regressions(terminal, cluster)
    combined = pd.concat([manual, regression], ignore_index=True, sort=False)

    manual_path = args.outdir / "model2_elasticity_manual.tsv"
    reg_path = args.outdir / "model2_elasticity_regression.tsv"
    comb_path = args.outdir / "model2_elasticity_combined.tsv"

    manual.to_csv(manual_path, sep="\t", index=False)
    regression.to_csv(reg_path, sep="\t", index=False)
    combined.to_csv(comb_path, sep="\t", index=False)

    print("=== Model_2_elasticity(v3): done ===")
    print(f"Terminal panel : {args.terminal_panel}")
    print(f"Cluster panel  : {args.cluster_panel}")
    print(f"Manual out     : {manual_path} (rows={len(manual)})")
    print(f"Regression out : {reg_path} (rows={len(regression)})")
    print(f"Combined out   : {comb_path} (rows={len(combined)})")


if __name__ == "__main__":
    main()
