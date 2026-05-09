#!/usr/bin/env python3
"""
Model_2_step2_elasticity_v9.py

Build the Model 2 elasticity layer.

v9 fixes
------------
1. Keeps manual legacy elasticity as the preferred default.
2. Adds richer metadata on whether an elasticity is preferred-manual,
   robustness-manual, regression-fallback, or unavailable.
3. Adds a preferred manual aggregate elasticity proxy for the Haifa port cluster,
   constructed from 2024 Haifa-specific HPC + IPC annual-report labor-share objects.
4. Still supports an optional user-supplied aggregate elasticity override when desired.

Outputs
-------
Design/Output (new)/Model_2_final/Tables/
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


def extract_logkl(res) -> Dict[str, object]:
    return {
        "eta": float(res.params.get("log_KL", np.nan)),
        "eta_se": float(res.bse.get("log_KL", np.nan)),
        "pvalue": float(res.pvalues.get("log_KL", np.nan)),
        "N": int(round(res.nobs)),
        "R2": float(res.rsquared),
    }


# ---------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------

def classify_eta_usage(row: Dict[str, object]) -> str:
    family = str(row.get("eta_family", ""))
    role = str(row.get("eta_role", ""))
    preferred = int(row.get("preferred_flag", 0) or 0)
    entity = str(row.get("entity", ""))

    if family == "manual" and entity == "Haifa port cluster" and preferred == 1:
        return "preferred_manual_aggregate"
    if family == "manual" and preferred == 1:
        return "preferred_manual"
    if family == "manual" and role == "robustness":
        return "manual_robustness"
    if family == "regression" and entity == "Haifa port cluster":
        return "aggregate_regression_fallback"
    if family == "regression":
        return "regression_robustness"
    if str(row.get("status", "")) == "unavailable":
        return "unavailable"
    return "diagnostic"


# ---------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------
# 2024 Haifa-specific annual-report inputs (million NIS)
HPC_2024 = {"labor_comp": 342.501, "dep_amort": 63.266, "op_surplus_proxy": 51.430}
IPC_2024 = {"labor_comp": 185.863, "dep_amort": 535.800, "op_surplus_proxy": 339.599}


def compute_aggregate_manual_proxy_hpc_ipc() -> Dict[str, float]:
    va_hpc = HPC_2024["labor_comp"] + HPC_2024["dep_amort"] + HPC_2024["op_surplus_proxy"]
    va_ipc = IPC_2024["labor_comp"] + IPC_2024["dep_amort"] + IPC_2024["op_surplus_proxy"]
    labor_total = HPC_2024["labor_comp"] + IPC_2024["labor_comp"]
    va_total = va_hpc + va_ipc
    labor_share = labor_total / va_total
    alpha = 1.0 - labor_share
    return {
        "va_hpc": va_hpc,
        "va_ipc": va_ipc,
        "labor_total": labor_total,
        "va_total": va_total,
        "labor_share": labor_share,
        "alpha": alpha,
    }



def build_manual_rows(aggregate_manual_eta: Optional[float], aggregate_manual_label: str, use_default_aggregate_manual: bool = True) -> pd.DataFrame:
    rows: List[Dict[str, object]] = [
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
    ]

    proxy = compute_aggregate_manual_proxy_hpc_ipc() if use_default_aggregate_manual else None

    if aggregate_manual_eta is not None and np.isfinite(aggregate_manual_eta):
        rows.append(
            {
                "entity": "Haifa port cluster",
                "eta_family": "manual",
                "eta_source": aggregate_manual_label,
                "eta_role": "preferred",
                "spec": "manual",
                "delta": np.nan,
                "eta": float(aggregate_manual_eta),
                "eta_se": np.nan,
                "pvalue": np.nan,
                "N": np.nan,
                "R2": np.nan,
                "preferred_flag": 1,
                "status": "ok",
                "reason": "user-supplied manual aggregate elasticity override",
                "notes": "Optional aggregate manual elasticity supplied by the user.",
            }
        )
    elif proxy is not None and np.isfinite(proxy["alpha"]):
        rows.append(
            {
                "entity": "Haifa port cluster",
                "eta_family": "manual",
                "eta_source": "Haifa_port_HPC_IPC_2024_proxy",
                "eta_role": "preferred",
                "spec": "manual",
                "delta": np.nan,
                "eta": float(proxy["alpha"]),
                "eta_se": np.nan,
                "pvalue": np.nan,
                "N": np.nan,
                "R2": np.nan,
                "preferred_flag": 1,
                "status": "ok",
                "reason": "default preferred manual aggregate proxy from 2024 HPC+IPC annual-report objects",
                "notes": (
                    f"HPC+IPC 2024 proxy: labor_total={proxy['labor_total']:.3f}, "
                    f"va_total={proxy['va_total']:.3f}, labor_share={proxy['labor_share']:.6f}, alpha={proxy['alpha']:.6f}."
                ),
            }
        )
    else:
        rows.append(
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
                "notes": "Aggregate manual elasticity intentionally left unspecified.",
            }
        )

    for row in rows:
        row["eta_usage_class"] = classify_eta_usage(row)
    return pd.DataFrame(rows)


def normalize_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    for c in ["log_LP", "log_KL", "t_index", "month_index", "quarter_index", "delta"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "t_index" not in df.columns:
        if "quarter_index" in df.columns:
            df["t_index"] = pd.Series(df["quarter_index"]).rank(method="dense")
        elif "month_index" in df.columns:
            df["t_index"] = pd.Series(df["month_index"]).rank(method="dense")
    return df.replace([np.inf, -np.inf], np.nan)


def estimate_regressions(term: pd.DataFrame, cluster: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

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

        for entity, source in [("Haifa--Legacy", "legacy_ts_trend"), ("Haifa--Bayport", "bayport_ts_trend")]:
            sub = term_d[term_d["entity"] == entity].dropna(subset=["log_LP", "log_KL", "t_index"]).copy()
            if len(sub) < 8:
                row = {
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
                }
                row["eta_usage_class"] = classify_eta_usage(row)
                rows.append(row)
                continue
            res = run_ols(sub, "log_LP ~ log_KL + t_index")
            row = {
                "entity": entity,
                "eta_family": "regression",
                "eta_source": source,
                "eta_role": "robustness",
                "spec": "TS+trend",
                "delta": delta,
                **extract_logkl(res),
                "preferred_flag": 0,
                "status": "ok",
                "reason": "HC1 time-series regression",
                "notes": "Reduced-form co-movement estimate, not structural.",
            }
            row["eta_usage_class"] = classify_eta_usage(row)
            rows.append(row)

        pooled = term_d.dropna(subset=["log_LP", "log_KL", "t_index", "entity"]).copy()
        if len(pooled) >= 12 and pooled["entity"].nunique() >= 2:
            res = run_ols(pooled, "log_LP ~ log_KL + t_index + C(entity)")
            row = {
                "entity": "Pooled Haifa terminals",
                "eta_family": "regression",
                "eta_source": "pooled_trend_fe",
                "eta_role": "robustness",
                "spec": "Pooled+trend+FE",
                "delta": delta,
                **extract_logkl(res),
                "preferred_flag": 0,
                "status": "ok",
                "reason": "HC1 pooled regression",
                "notes": "Pooled terminal regression with entity fixed effects and linear time trend.",
            }
        else:
            row = {
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
            }
        row["eta_usage_class"] = classify_eta_usage(row)
        rows.append(row)

        cl = cluster_d.dropna(subset=["log_LP", "log_KL", "t_index"]).copy() if not cluster_d.empty else cluster_d.copy()
        if len(cl) >= 8:
            res = run_ols(cl, "log_LP ~ log_KL + t_index")
            row = {
                "entity": "Haifa port cluster",
                "eta_family": "regression",
                "eta_source": "cluster_ts_trend",
                "eta_role": "robustness",
                "spec": "TS+trend",
                "delta": delta,
                **extract_logkl(res),
                "preferred_flag": 0,
                "status": "ok",
                "reason": "HC1 aggregate time-series regression",
                "notes": "Aggregate-port robustness regression only.",
            }
        else:
            row = {
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
            }
        row["eta_usage_class"] = classify_eta_usage(row)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    thesis_root = find_thesis_root()

    parser = argparse.ArgumentParser(description="Build the current Model 2 elasticity layer.")
    preferred_terminal = thesis_root / "Design" / "Output (new)" / "Model_2_final" / "Inputs" / "model2_terminal_panel_quarterly.tsv"
    fallback_terminal = thesis_root / "Design" / "Output (new)" / "Model_2_final" / "Inputs" / "model2_terminal_panel.tsv"
    parser.add_argument("--terminal_panel", type=Path, default=preferred_terminal if preferred_terminal.exists() else fallback_terminal)
    parser.add_argument("--cluster_panel", type=Path, default=thesis_root / "Design" / "Output (new)" / "Model_2_final" / "Inputs" / "model2_cluster_panel.tsv")
    parser.add_argument("--outdir", type=Path, default=thesis_root / "Design" / "Output (new)" / "Model_2_final" / "Tables")
    parser.add_argument("--aggregate_manual_eta", type=float, default=None, help="Optional manual aggregate elasticity override.")
    parser.add_argument("--aggregate_manual_label", type=str, default="manual_aggregate_override")
    parser.add_argument("--no_default_aggregate_manual", action="store_true", help="Do not auto-build the default HPC+IPC aggregate manual proxy.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.terminal_panel.exists():
        raise FileNotFoundError(f"Terminal panel not found: {args.terminal_panel}")
    quarterly_sibling = args.terminal_panel.with_name("model2_terminal_panel_quarterly.tsv")
    if args.terminal_panel.name == "model2_terminal_panel.tsv" and quarterly_sibling.exists():
        print(f"[INFO] Using quarterly terminal panel for elasticity estimation: {quarterly_sibling}")
        args.terminal_panel = quarterly_sibling

    terminal = normalize_panel(args.terminal_panel)
    cluster = normalize_panel(args.cluster_panel) if args.cluster_panel.exists() else pd.DataFrame(columns=terminal.columns)

    manual = build_manual_rows(args.aggregate_manual_eta, args.aggregate_manual_label, use_default_aggregate_manual=not args.no_default_aggregate_manual)
    regression = estimate_regressions(terminal, cluster)
    combined = pd.concat([manual, regression], ignore_index=True, sort=False)

    manual_path = args.outdir / "model2_elasticity_manual.tsv"
    reg_path = args.outdir / "model2_elasticity_regression.tsv"
    comb_path = args.outdir / "model2_elasticity_combined.tsv"

    manual.to_csv(manual_path, sep="\t", index=False)
    regression.to_csv(reg_path, sep="\t", index=False)
    combined.to_csv(comb_path, sep="\t", index=False)

    print("=== Model_2_step2_elasticity_v9.py: done ===")
    print(f"Terminal panel         : {args.terminal_panel}")
    print(f"Cluster panel          : {args.cluster_panel}")
    agg_row = manual.loc[manual["entity"].astype(str) == "Haifa port cluster"].copy()
    effective_agg = agg_row["eta"].iloc[0] if not agg_row.empty else np.nan
    effective_source = agg_row["eta_source"].iloc[0] if not agg_row.empty else ""
    print(f"Aggregate manual eta   : {args.aggregate_manual_eta}")
    print(f"Default aggregate proxy: {"off" if args.no_default_aggregate_manual else "HPC+IPC 2024"}")
    print(f"Effective aggregate eta: {effective_agg}")
    print(f"Effective aggregate src: {effective_source}")
    print(f"Manual out             : {manual_path} (rows={len(manual)})")
    print(f"Regression out         : {reg_path} (rows={len(regression)})")
    print(f"Combined out           : {comb_path} (rows={len(combined)})")


if __name__ == "__main__":
    main()
