#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model_2_mediation.py

Mediation-style decomposition of reform effects on log(LP).

This script combines:
  - Total effect TE from Model 1A (dynamic event-study betas on log(LP))
  - Capital-deepening effect ΔC from Model 1B window betas on log(K/L)
    (terminal-level from Model_1B_relaxed, cluster-level from Model_1B_Haifa)
  - Elasticity η from Model 2 (log(LP) vs log(K/L))

and reports:
  ME = η * ΔC,  RE = TE - ME,  s = ME/TE.

Key change: privatization horizons
---------------------------------
Haifa privatization event date is 2023-01 and the panels end 2024-12, so the
maximum post horizon is m=23 (2024-12).

We therefore support TWO privatization reform IDs:
  - reform='haifa_priv'      : pre-war window (max_post=8, through 2023-09)
  - reform='haifa_priv_long' : extended window (max_post=23, through 2024-12)

Critical guardrail:
-------------------
We do NOT silently truncate windows. If a requested [a,b] horizon is not fully
supported by the available event-time betas, the script returns NaNs for that
cell.

Outputs
-------
Design/Output (new)/Model_2/Tables/
  - model2_mediation_main.tsv
  - model2_mediation_appendix.tsv

The "main" file keeps the pooled Haifa port-cluster decomposition for the main
reforms (haifa_comp and haifa_priv_long). The appendix file includes terminal-
by-terminal, pooled-terminals, and cluster rows for *all* included reforms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Reform plans + horizons
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ReformPlan:
    reform: str
    label: str
    event_year: int
    event_month: int
    max_post: int
    is_main: bool


REFORMS: List[ReformPlan] = [
    ReformPlan(
        reform="haifa_comp",
        label="Haifa competition entry",
        event_year=2021,
        event_month=9,
        max_post=24,
        is_main=True,
    ),
    ReformPlan(
        reform="haifa_priv",
        label="Haifa privatization (pre-war)",
        event_year=2023,
        event_month=1,
        max_post=8,
        is_main=False,
    ),
    ReformPlan(
        reform="haifa_priv_long",
        label="Haifa privatization (extended)",
        event_year=2023,
        event_month=1,
        max_post=23,
        is_main=True,
    ),
]


@dataclass(frozen=True)
class Horizon:
    key: str              # 'post_all' / 'post_y1' / 'post_y2'
    a: int
    b: int
    label: str            # '[1,23]' etc
    kl_window_name: str   # matches Model 1B window_name


def horizons_for(plan: ReformPlan) -> List[Horizon]:
    out: List[Horizon] = []

    # post_all on true support
    out.append(Horizon("post_all", 1, plan.max_post, f"[1,{plan.max_post}]", "post_all"))

    # year 1
    y1_end = min(12, plan.max_post)
    out.append(Horizon("post_y1", 1, y1_end, f"[1,{y1_end}]", "post_y1"))

    # year 2 (if supported)
    if plan.max_post >= 13:
        out.append(Horizon("post_y2", 13, plan.max_post, f"[13,{plan.max_post}]", "post_y2"))

    return out


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def find_thesis_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if p.name.lower() == "thesis":
            return p
    return here.parents[2]


THESIS_ROOT = find_thesis_root()
OUT_DIR = THESIS_ROOT / "Design" / "Output (new)" / "Model_2" / "Tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATHS = {
    # Model 1A (dynamic betas)
    "m1a_dyn": THESIS_ROOT / "Design" / "Output (new)" / "Model_1A" / "model1a_lp_dynamic_betas_all.tsv",
    # Model 1B window betas
    "m1b_term_win": THESIS_ROOT / "Design" / "Output (new)" / "Model_1B_relaxed" / "model1b_relaxed_window_betas.tsv",
    "m1b_cluster_win": THESIS_ROOT / "Design" / "Output (new)" / "Model_1B_Haifa" / "model1b_haifa_window_betas.tsv",
    # Model 2 elasticities
    "eta_term": THESIS_ROOT / "Design" / "Output (new)" / "Model_2" / "Tables" / "model2_elasticity_results.tsv",
    "eta_cluster": THESIS_ROOT / "Design" / "Output (new)" / "Model_2" / "Tables" / "model2_cluster_elasticity_results.tsv",
    # LP panel (for pooling weights)
    "lp_panel": THESIS_ROOT / "Data" / "LP" / "LP_Panel_monthly.tsv",
}


# ---------------------------------------------------------------------
# IO + normalization
# ---------------------------------------------------------------------

def read_tsv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    return pd.read_csv(path, sep="\t")


def ensure_cols(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


def normalize_win_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Model 1B window-beta tables to have (spec_name,beta_hat,se)."""
    out = df.copy()

    if "spec_name" not in out.columns and "fe_type" in out.columns:
        out = out.rename(columns={"fe_type": "spec_name"})

    # normalize coefficient column name
    if "beta_hat" not in out.columns:
        if "beta" in out.columns:
            out = out.rename(columns={"beta": "beta_hat"})
        elif "coef" in out.columns:
            out = out.rename(columns={"coef": "beta_hat"})

    if "se" not in out.columns:
        if "stderr" in out.columns:
            out = out.rename(columns={"stderr": "se"})
        elif "std_err" in out.columns:
            out = out.rename(columns={"std_err": "se"})

    return out


def normalize_eta_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Model 2 elasticity tables to have (entity,delta,spec,eta,eta_se)."""
    out = df.copy()

    # Ensure standard key names
    if "entity" not in out.columns and "target" in out.columns:
        out = out.rename(columns={"target": "entity"})

    # Coef column
    if "eta" not in out.columns:
        for c in ["coef_log_KL", "coef", "coefficient", "beta", "estimate"]:
            if c in out.columns:
                out = out.rename(columns={c: "eta"})
                break

    # SE column
    if "eta_se" not in out.columns:
        for c in ["se_log_KL", "se", "stderr", "std_err", "std_error"]:
            if c in out.columns:
                out = out.rename(columns={c: "eta_se"})
                break

    return out


# ---------------------------------------------------------------------
# TE (Model 1A) aggregation from dynamic betas
# ---------------------------------------------------------------------

def te_window_from_dynamic(
    dyn: pd.DataFrame,
    reform: str,
    target: str,
    spec_name: str,
    a: int,
    b: int,
    require_full: bool = True,
) -> Tuple[float, float, int]:
    """
    Compute TE[a,b] as the equal-weight mean of event-time betas for m in [a,b].

    SE uses an independence approximation: Var(mean) = sum(se_m^2) / k^2.

    Returns (beta_hat, se, k_months_used). If require_full is True, returns NaN
    if any month in [a,b] is missing.
    """
    sub = dyn[(dyn["reform"] == reform) & (dyn["target"] == target) & (dyn["spec_name"] == spec_name)].copy()
    if sub.empty:
        return (np.nan, np.nan, 0)

    sub["event_time"] = sub["event_time"].astype(int)
    win = sub[(sub["event_time"] >= a) & (sub["event_time"] <= b)].copy()

    expected = max(0, b - a + 1)
    k = len(win)

    if require_full and (k != expected):
        return (np.nan, np.nan, k)
    if k == 0:
        return (np.nan, np.nan, 0)

    beta_hat = float(win["beta"].mean())
    se = float(np.sqrt(np.nansum(np.square(win["se"].values))) / k)
    return (beta_hat, se, k)


# ---------------------------------------------------------------------
# TEU weights for pooling (LP panel)
# ---------------------------------------------------------------------

def teu_weights_for_horizon(
    lp: pd.DataFrame,
    plan: ReformPlan,
    a: int,
    b: int,
    legacy_series: str = "Haifa_Legacy_Q",
    sipg_series: str = "Haifa_SIPG_Q",
) -> Dict[str, float]:
    """Compute TEU weights for pooling Legacy vs SIPG over horizon [a,b]."""
    if not {"series_id", "year", "month", "TEU"}.issubset(lp.columns):
        return {legacy_series: 0.5, sipg_series: 0.5}

    df = lp[lp["series_id"].isin([legacy_series, sipg_series])].copy()
    if df.empty:
        return {legacy_series: 0.5, sipg_series: 0.5}

    df["event_time"] = (
        (df["year"].astype(int) - int(plan.event_year)) * 12
        + (df["month"].astype(int) - int(plan.event_month))
    )

    win = df[(df["event_time"] >= a) & (df["event_time"] <= b)].copy()
    if win.empty:
        return {legacy_series: 0.5, sipg_series: 0.5}

    sums = win.groupby("series_id")["TEU"].sum(min_count=1)
    tot = float(sums.sum())
    if not np.isfinite(tot) or tot <= 0:
        return {legacy_series: 0.5, sipg_series: 0.5}

    wL = float(sums.get(legacy_series, 0.0) / tot)
    wB = float(sums.get(sipg_series, 0.0) / tot)
    if (wL == 0.0) and (wB == 0.0):
        return {legacy_series: 0.5, sipg_series: 0.5}

    return {legacy_series: wL, sipg_series: wB}


# ---------------------------------------------------------------------
# ΔC (Model 1B) window lookup
# ---------------------------------------------------------------------

def pick_window_beta(
    win_df: pd.DataFrame,
    reform: str,
    target: str,
    spec_name: Optional[str],
    window_name: str,
    required_m: Tuple[int, int],
) -> Tuple[float, float]:
    """Pick (beta_hat,se) for a given (reform,target,spec,window) with exact bounds."""
    df = win_df.copy()
    df = df[(df["reform"] == reform) & (df["target"] == target) & (df["window_name"] == window_name)].copy()
    if spec_name is not None and ("spec_name" in df.columns):
        df = df[df["spec_name"] == spec_name]

    if df.empty:
        return (np.nan, np.nan)

    ms, me = required_m
    df["m_start"] = df["m_start"].astype(int)
    df["m_end"] = df["m_end"].astype(int)
    df = df[(df["m_start"] == int(ms)) & (df["m_end"] == int(me))]

    if df.empty:
        return (np.nan, np.nan)

    row = df.iloc[0]
    return (float(row["beta_hat"]), float(row["se"]))


# ---------------------------------------------------------------------
# η lookup
# ---------------------------------------------------------------------

def pick_eta(df_eta: pd.DataFrame, entity: str, delta: float, spec: str) -> Tuple[float, float]:
    sub = df_eta[(df_eta["entity"] == entity) & (df_eta["delta"] == delta) & (df_eta["spec"] == spec)].copy()
    if sub.empty:
        return (np.nan, np.nan)
    r = sub.iloc[0]
    return (float(r["eta"]), float(r.get("eta_se", np.nan)))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=== Model_2_mediation: starting ===")
    print("THESIS_ROOT:", THESIS_ROOT)

    # Load inputs
    m1a_dyn = read_tsv(PATHS["m1a_dyn"])
    m1b_term = read_tsv(PATHS["m1b_term_win"])
    m1b_cluster = read_tsv(PATHS["m1b_cluster_win"])
    eta_term = read_tsv(PATHS["eta_term"])
    eta_cluster = read_tsv(PATHS["eta_cluster"])
    lp_panel = read_tsv(PATHS["lp_panel"])

    if m1a_dyn is None or eta_term is None or eta_cluster is None or lp_panel is None:
        print("[ERROR] Missing required inputs; cannot run.")
        return

    # Normalize Model 1A dynamic
    m1a_dyn = m1a_dyn.copy()
    if "beta" not in m1a_dyn.columns and "beta_hat" in m1a_dyn.columns:
        m1a_dyn = m1a_dyn.rename(columns={"beta_hat": "beta"})
    if "se" not in m1a_dyn.columns and "stderr" in m1a_dyn.columns:
        m1a_dyn = m1a_dyn.rename(columns={"stderr": "se"})

    ensure_cols(m1a_dyn, ["reform", "target", "spec_name", "event_time", "beta", "se"], "Model1A dynamic")

    # Normalize Model 1B window tables
    if m1b_term is not None:
        m1b_term = normalize_win_df(m1b_term)
    if m1b_cluster is not None:
        m1b_cluster = normalize_win_df(m1b_cluster)

    # Normalize eta tables
    eta_term = normalize_eta_df(eta_term)
    eta_cluster = normalize_eta_df(eta_cluster)

    ensure_cols(eta_term, ["entity", "delta", "spec", "eta"], "eta_term")
    ensure_cols(eta_cluster, ["entity", "delta", "spec", "eta"], "eta_cluster")

    # Specs to use
    TE_SPEC = "porttr"
    KL_TERM_SPEC: Optional[str] = None
    KL_CLUSTER_SPEC: Optional[str] = "ts_trend"
    ETA_SPEC_TERM = "preferred"
    ETA_SPEC_CLUSTER = "preferred"
    DELTA = 0.06

    # Entity labels in elasticity outputs
    ETA_ENTITY_LEGACY = "Haifa--Legacy"
    ETA_ENTITY_BAYPORT = "Haifa--Bayport"
    ETA_ENTITY_POOLED = "Pooled Haifa terminals"
    ETA_ENTITY_CLUSTER = "Haifa port cluster"

    # Targets in Model 1B window tables
    KL_TARGET_LEGACY = "Haifa-Legacy K/L"
    KL_TARGET_BAYPORT = "Haifa-Bayport K/L (SIPG central)"
    KL_TARGET_CLUSTER = "Haifa port cluster"

    # Targets in Model 1A dynamic betas
    TE_TARGET_LEGACY = "Haifa-Legacy terminal"
    TE_TARGET_BAYPORT = "Haifa-Bayport terminal"

    # η lookups (constant across reforms/horizons)
    etaL, etaL_se = pick_eta(eta_term, ETA_ENTITY_LEGACY, DELTA, ETA_SPEC_TERM)
    etaB, etaB_se = pick_eta(eta_term, ETA_ENTITY_BAYPORT, DELTA, ETA_SPEC_TERM)
    etaP, etaP_se = pick_eta(eta_term, ETA_ENTITY_POOLED, DELTA, ETA_SPEC_TERM)
    etaC, etaC_se = pick_eta(eta_cluster, ETA_ENTITY_CLUSTER, DELTA, ETA_SPEC_CLUSTER)

    def med(te: float, dc: float, eta: float) -> Tuple[float, float, float]:
        me = eta * dc if np.isfinite(eta) and np.isfinite(dc) else np.nan
        re = te - me if np.isfinite(te) and np.isfinite(me) else np.nan
        s = me / te if np.isfinite(me) and np.isfinite(te) and te != 0 else np.nan
        return me, re, s

    main_rows: List[Dict] = []
    app_rows: List[Dict] = []

    for plan in REFORMS:
        for hz in horizons_for(plan):

            # TE windows from dynamic betas
            teL, teL_se, _ = te_window_from_dynamic(
                m1a_dyn, plan.reform, TE_TARGET_LEGACY, TE_SPEC, hz.a, hz.b, require_full=True
            )
            teB, teB_se, _ = te_window_from_dynamic(
                m1a_dyn, plan.reform, TE_TARGET_BAYPORT, TE_SPEC, hz.a, hz.b, require_full=True
            )

            # Pooling weights (TEU shares)
            w = teu_weights_for_horizon(lp_panel, plan, hz.a, hz.b)
            wL = w.get("Haifa_Legacy_Q", 0.5)
            wB = w.get("Haifa_SIPG_Q", 0.5)

            teP = wL * teL + wB * teB
            teP_se = float(np.sqrt((wL ** 2) * (teL_se ** 2) + (wB ** 2) * (teB_se ** 2))) if np.isfinite(teL_se) and np.isfinite(teB_se) else np.nan

            # ΔC (terminal)
            dCL = dCL_se = np.nan
            dCB = dCB_se = np.nan
            if m1b_term is not None:
                dCL, dCL_se = pick_window_beta(m1b_term, plan.reform, KL_TARGET_LEGACY, KL_TERM_SPEC, hz.kl_window_name, (hz.a, hz.b))
                dCB, dCB_se = pick_window_beta(m1b_term, plan.reform, KL_TARGET_BAYPORT, KL_TERM_SPEC, hz.kl_window_name, (hz.a, hz.b))

            # ΔC (cluster)
            dCC = dCC_se = np.nan
            if m1b_cluster is not None:
                dCC, dCC_se = pick_window_beta(m1b_cluster, plan.reform, KL_TARGET_CLUSTER, KL_CLUSTER_SPEC, hz.kl_window_name, (hz.a, hz.b))

            # pooled terminal ΔC (if both exist)
            dCP = (wL * dCL + wB * dCB) if np.isfinite(dCL) and np.isfinite(dCB) else np.nan

            # Mediation arithmetic
            meL, reL, sL = med(teL, dCL, etaL)
            meB, reB, sB = med(teB, dCB, etaB)
            meP, reP, sP = med(teP, dCP, etaP)
            meC, reC, sC = med(teP, dCC, etaC)

            # Appendix rows
            app_rows.extend([
                {
                    "entity": "Haifa--Legacy",
                    "reform": plan.reform,
                    "reform_label": plan.label,
                    "horizon": hz.label,
                    "a": hz.a, "b": hz.b,
                    "TE": teL, "TE_se": teL_se,
                    "dC": dCL, "dC_se": dCL_se,
                    "eta": etaL, "eta_se": etaL_se,
                    "ME": meL, "RE": reL, "share": sL,
                    "delta": DELTA, "spec_te": TE_SPEC, "spec_kl": str(KL_TERM_SPEC), "spec_eta": ETA_SPEC_TERM,
                },
                {
                    "entity": "Haifa--Bayport",
                    "reform": plan.reform,
                    "reform_label": plan.label,
                    "horizon": hz.label,
                    "a": hz.a, "b": hz.b,
                    "TE": teB, "TE_se": teB_se,
                    "dC": dCB, "dC_se": dCB_se,
                    "eta": etaB, "eta_se": etaB_se,
                    "ME": meB, "RE": reB, "share": sB,
                    "delta": DELTA, "spec_te": TE_SPEC, "spec_kl": str(KL_TERM_SPEC), "spec_eta": ETA_SPEC_TERM,
                },
                {
                    "entity": "Pooled Haifa terminals",
                    "reform": plan.reform,
                    "reform_label": plan.label,
                    "horizon": hz.label,
                    "a": hz.a, "b": hz.b,
                    "TE": teP, "TE_se": teP_se,
                    "dC": dCP, "dC_se": np.nan,
                    "eta": etaP, "eta_se": etaP_se,
                    "ME": meP, "RE": reP, "share": sP,
                    "delta": DELTA, "spec_te": TE_SPEC, "spec_kl": "pooled_terminals", "spec_eta": ETA_SPEC_TERM,
                },
                {
                    "entity": "Haifa port cluster",
                    "reform": plan.reform,
                    "reform_label": plan.label,
                    "horizon": hz.label,
                    "a": hz.a, "b": hz.b,
                    "TE": teP, "TE_se": teP_se,
                    "dC": dCC, "dC_se": dCC_se,
                    "eta": etaC, "eta_se": etaC_se,
                    "ME": meC, "RE": reC, "share": sC,
                    "delta": DELTA, "spec_te": TE_SPEC, "spec_kl": KL_CLUSTER_SPEC, "spec_eta": ETA_SPEC_CLUSTER,
                },
            ])

            # Main table: pooled cluster mediation, only main plans
            if plan.is_main:
                main_rows.append(
                    {
                        "reform": plan.reform,
                        "reform_label": plan.label,
                        "horizon": hz.label,
                        "a": hz.a, "b": hz.b,
                        "TE": teP, "TE_se": teP_se,
                        "dC": dCC, "dC_se": dCC_se,
                        "eta": etaC, "eta_se": etaC_se,
                        "ME": meC, "RE": reC, "share": sC,
                        "delta": DELTA,
                        "spec_te": TE_SPEC,
                        "spec_kl": KL_CLUSTER_SPEC,
                        "spec_eta": ETA_SPEC_CLUSTER,
                    }
                )

    main_df = pd.DataFrame(main_rows)
    app_df = pd.DataFrame(app_rows)

    out_main = OUT_DIR / "model2_mediation_main.tsv"
    out_app = OUT_DIR / "model2_mediation_appendix.tsv"

    main_df.to_csv(out_main, sep="\t", index=False)
    app_df.to_csv(out_app, sep="\t", index=False)

    print(f"Wrote main table TSV: {out_main} (rows={len(main_df)})")
    print(f"Wrote appendix TSV: {out_app} (rows={len(app_df)})")
    print("=== Model_2_mediation: done ===")


if __name__ == "__main__":
    main()
