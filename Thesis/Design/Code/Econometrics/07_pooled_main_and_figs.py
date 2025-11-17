#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_pooled_main_and_figs.py  (fixed)
------------------------------------
Step 07 — pooled entrant/legacy results:
- Reads Step 05 panel (terminal-level, shared-pre, with tau bins).
- Pools terminals into roles {entrant, legacy}.
- Runs role-specific dynamic ES with terminal FE + calendar FE.
- Builds a "Table 1" scalar: average post effect over a window (default k=+1..+4).
- Plots dynamic paths with N per bin and pretrend p-value.
- Writes CSVs + PNGs + meta JSON into Design/Output Data.

Fixes in this version:
- Correctly index the per-point upper CI when placing N labels (previously passed a vector).
- Suppress non-fatal statsmodels constraint-rank warnings from F-tests.
"""

# ====================== USER TOGGLE =======================
INPUT_CANDIDATES = [
    "Design/Output Data/05_panel_terminal_sharedpre_model1.csv",
    "Design/Output Data/panel_terminal_sharedpre_model1.csv",
]
OUTDIR = "Design/Output Data"
POST_WINDOW = (1, 4)   # inclusive event-time window for Table 1
EXCLUDE_QTRS = []      # e.g. ["2020Q2","2020Q3"]
TERMINAL_MAP = {"SIPG": "entrant", "HCT": "entrant", "Legacy": "legacy"}
# ==========================================================

import os, sys, json, itertools, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Silence non-fatal constraint-rank warnings in joint F-tests
warnings.filterwarnings("ignore", message="covariance of constraints does not have full rank")

def ensure_outdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def find_input(paths):
    for p in paths:
        if Path(p).exists():
            return p
    sys.exit(f"[07] Could not find Step-05 input. Tried: {paths}")

def build_role(df: pd.DataFrame) -> pd.Series:
    return df["terminal"].map(TERMINAL_MAP)

def _design_matrix_with_fe(df: pd.DataFrame, add_cols: list) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Terminal FE + calendar FE + custom columns in add_cols."""
    gdf = df.copy()
    gdf["tid"] = gdf["port"].astype(str) + ":" + gdf["terminal"].astype(str)
    tid_d = pd.get_dummies(gdf["tid"], prefix="tid", drop_first=True)
    qtr_d = pd.get_dummies(gdf["qtr"], prefix="qtr", drop_first=True)
    X = pd.concat([tid_d, qtr_d, gdf[add_cols]], axis=1)
    # numeric & non-constant guards
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.loc[:, X.sum(axis=0) != 0]
    X = X.loc[:, X.var(axis=0) > 0]
    y = pd.to_numeric(gdf["Y"], errors="coerce").astype(float)
    valid = ~(y.isna() | X.isna().any(axis=1))
    return X.loc[valid, :], y.loc[valid].values, gdf.loc[valid, "port"].values

def fit_dynamic(df: pd.DataFrame, role_value: str):
    gdf = df.copy()
    gdf["role"] = build_role(gdf)
    gdf = gdf[gdf["role"].isin(["entrant","legacy"])].copy()
    gdf["is_role"] = (gdf["role"] == role_value).astype(int)
    # build interaction dummies 1{tau=k} * 1{role=role_value}
    bins_all = sorted(pd.to_numeric(gdf["tau_bin"], errors="coerce").dropna().astype(int).unique().tolist())
    pre_bins = [k for k in bins_all if k < 0]
    tau = pd.to_numeric(gdf["tau_bin"], errors="coerce").astype("Int64")
    Dcols, klist = [], []
    for k in bins_all:
        col = f"D_{k}"
        gdf[col] = ((tau == k).astype(int) * gdf["is_role"]).astype(float)
        Dcols.append(col); klist.append(k)

    X, y, clusters = _design_matrix_with_fe(gdf, Dcols)
    if X.shape[0] <= X.shape[1] or X.empty:
        return pd.DataFrame(), np.nan, {"n_obs": int(X.shape[0]) if not X.empty else 0}
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": clusters})

    rows = []
    for k, col in zip(klist, Dcols):
        if col in res.params.index:
            b, se = float(res.params[col]), float(res.bse[col])
            ci_lo, ci_hi = (b - 1.96*se, b + 1.96*se)
            n_k = int(((tau == k) & (gdf["is_role"] == 1)).sum())
            rows.append({"role": role_value, "k": k, "beta": b, "se": se,
                         "ci_lo": ci_lo, "ci_hi": ci_hi, "n_k": n_k})
    coef_df = pd.DataFrame(rows).sort_values("k")

    # pretrend joint F test on lead bins
    pre_cols = [f"D_{k}" for k in pre_bins if f"D_{k}" in res.params.index]
    pre_p = None
    if pre_cols:
        R = np.zeros((len(pre_cols), len(res.params)))
        param_index = list(res.params.index)
        for r, name in enumerate(pre_cols):
            if name in param_index:
                R[r, param_index.index(name)] = 1.0
        try:
            pre_p = float(res.f_test(R).pvalue)
        except Exception:
            pre_p = None
    return coef_df, pre_p, {"n_obs": int(X.shape[0])}

def wild_cluster_exact_two(y, X, clusters, param_name: str):
    """Exact Rademacher wild-cluster bootstrap for 2 clusters."""
    unique = pd.Series(clusters).astype(str).unique().tolist()
    if len(unique) != 2:
        return np.nan
    base = sm.OLS(y, X).fit()
    resid = y - X @ base.params.values
    idx = list(base.params.index).index(param_name)
    bhat = float(base.params[idx])
    import itertools
    signs = list(itertools.product([-1,1], repeat=2))
    betas = []
    for s0, s1 in signs:
        w = np.ones_like(resid)
        w[np.array(clusters, dtype=str) == unique[0]] *= s0
        w[np.array(clusters, dtype=str) == unique[1]] *= s1
        yb = (X @ base.params.values) + resid * w
        b = sm.OLS(yb, X).fit().params[idx]
        betas.append(float(b))
    return float(np.mean(np.abs(betas) >= abs(bhat)))

def fit_scalar_avgpost(df: pd.DataFrame, role_value: str, post_window=(1,4)):
    gdf = df.copy()
    gdf["role"] = build_role(gdf)
    gdf = gdf[gdf["role"].isin(["entrant","legacy"])].copy()
    gdf["is_role"] = (gdf["role"] == role_value).astype(int)
    tau = pd.to_numeric(gdf["tau_bin"], errors="coerce").astype("Int64")
    lo, hi = post_window
    gdf["PostAvg_role"] = (((tau >= lo) & (tau <= hi)).astype(int) * gdf["is_role"]).astype(float)
    X, y, clusters = _design_matrix_with_fe(gdf, ["PostAvg_role"])
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    b, se, p = float(res.params["PostAvg_role"]), float(res.bse["PostAvg_role"]), float(res.pvalues["PostAvg_role"])
    try:
        p_wild = wild_cluster_exact_two(y, X, clusters, "PostAvg_role")
    except Exception:
        p_wild = np.nan
    return {"role": role_value, "coef": b, "se": se, "p_crv": p, "p_wild": p_wild, "N": int(X.shape[0])}

def plot_dynamic(coef_df: pd.DataFrame, pre_p: float, role_value: str, outpath: str):
    plt.figure(figsize=(7.5,4.8))
    xs = coef_df["k"].values
    ys = coef_df["beta"].values
    ylo = coef_df["ci_lo"].values
    yhi = coef_df["ci_hi"].values
    ns  = coef_df["n_k"].values
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.errorbar(xs, ys, yerr=[ys - ylo, yhi - ys], fmt="o", capsize=3)
    # Place N labels above the upper CI per point
    for i, (x, y, n) in enumerate(zip(xs, ys, ns)):
        y_upper = yhi[i]
        plt.annotate(f"N={n}", (x, y_upper), xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=9)
    plt.title(f"Event Study — {role_value.capitalize()} (pooled, baseline k=-1 omitted)")
    plt.xlabel("Event time (quarters)")
    plt.ylabel("Effect on ln(LP)")
    if pre_p is not None and not np.isnan(pre_p):
        plt.text(0.98, 0.02, f"Pre-trends p = {pre_p:.3f}", ha="right", va="bottom", transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def main():
    inp = find_input(INPUT_CANDIDATES)
    ensure_outdir(OUTDIR)
    df = pd.read_csv(inp)
    need = {"Y","port","terminal","qtr","t_index","tau_bin"}
    miss = need - set(df.columns)
    if miss:
        sys.exit(f"[07] Missing columns: {sorted(miss)}")
    if EXCLUDE_QTRS:
        df = df[~df["qtr"].isin(EXCLUDE_QTRS)].copy()
    df["role"] = build_role(df)
    df = df[df["role"].isin(["entrant","legacy"])].copy()

    # Dynamic ES
    es_ent, pre_ent, info_ent = fit_dynamic(df, "entrant")
    es_leg, pre_leg, info_leg = fit_dynamic(df, "legacy")
    (Path(OUTDIR)/"07_es_coeffs_entrant.csv").write_text(es_ent.to_csv(index=False), encoding="utf-8")
    (Path(OUTDIR)/"07_es_coeffs_legacy.csv").write_text(es_leg.to_csv(index=False), encoding="utf-8")
    if not es_ent.empty:
        plot_dynamic(es_ent, pre_ent, "entrant", str(Path(OUTDIR)/"07_fig_es_entrant.png"))
    if not es_leg.empty:
        plot_dynamic(es_leg, pre_leg, "legacy", str(Path(OUTDIR)/"07_fig_es_legacy.png"))

    # Table 1: scalar "average post"
    row_ent = fit_scalar_avgpost(df, "entrant", POST_WINDOW)
    row_leg = fit_scalar_avgpost(df, "legacy", POST_WINDOW)
    table = pd.DataFrame([row_ent, row_leg])[["role","coef","se","p_crv","p_wild","N"]]
    table["post_window"] = f"[{POST_WINDOW[0]},{POST_WINDOW[1]}]"
    (Path(OUTDIR)/"07_table_main.csv").write_text(table.to_csv(index=False), encoding="utf-8")

    meta = {
        "input": inp,
        "outdir": OUTDIR,
        "post_window": POST_WINDOW,
        "exclude_qtrs": EXCLUDE_QTRS,
        "pretrend_p": {"entrant": pre_ent, "legacy": pre_leg},
        "n_obs": {"entrant": info_ent.get("n_obs", None), "legacy": info_leg.get("n_obs", None)},
        "artifacts": {
            "coeffs_entrant_csv": "Design/Output Data/07_es_coeffs_entrant.csv",
            "coeffs_legacy_csv": "Design/Output Data/07_es_coeffs_legacy.csv",
            "fig_es_entrant_png": "Design/Output Data/07_fig_es_entrant.png",
            "fig_es_legacy_png": "Design/Output Data/07_fig_es_legacy.png",
            "table_main_csv": "Design/Output Data/07_table_main.csv",
        }
    }
    Path(Path(OUTDIR)/"07__meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[07] Done")

if __name__ == "__main__":
    here = Path(__file__).resolve()
    try:
        os.chdir(here.parents[3])  # project root (…/Thesis)
    except Exception:
        os.chdir(here.parent)
    main()
