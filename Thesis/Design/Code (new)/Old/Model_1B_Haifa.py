#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model_1B_Haifa.py

Haifa *port-cluster* log(K/L) time-series window regressions.

This script estimates coarse window-average ΔC for the Haifa port cluster
K/L series (series_id == 'Haifa_port_KL_cluster_central'). These estimates are
used downstream by Model_2_mediation.py.

Windows (in event-time months)
------------------------------
We report the familiar window names:
  - pre_all  : m in [-12,-2]
  - post_y1  : m in [1,min(12,max_post)]
  - post_y2  : m in [13,max_post] (only if max_post >= 13)
  - post_all : m in [1,max_post] (computed as a month-weighted average of post_y1
               and post_y2 coefficients)

Privatization: pre-war + extended specs
---------------------------------------
Haifa privatization event date is 2023-01. With data through 2024-12, the
maximum feasible post horizon is m=23 (2024-12). We therefore include:
  - reform='haifa_priv'       : pre-war sample with max_post = 8 (through 2023-09)
  - reform='haifa_priv_long'  : extended sample with max_post = 23 (through 2024-12)

Outputs
-------
THESIS_ROOT/Design/Output (new)/Model_1B_Haifa/
  - model1b_haifa_window_betas.tsv
  - model1b_haifa_pretrend_tests.tsv
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------

MODEL_NAME = "Model_1B_Haifa"
CLUSTER_SERIES_ID = "Haifa_port_KL_cluster_central"

# Keep consistent with other scripts
MIN_EVENT_TIME = -12


# ---------------------------------------------------------------------
# Reform configs
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ReformConfig:
    reform: str
    event_year: int
    event_month: int
    max_post: int
    label: str


REFORMS: List[ReformConfig] = [
    ReformConfig(
        reform="haifa_comp",
        event_year=2021,
        event_month=9,
        max_post=24,
        label="Haifa competition entry (2021-09; post<=24)",
    ),
    # Privatization pre-war: keep the original sample end (2023-09) => m=8
    ReformConfig(
        reform="haifa_priv",
        event_year=2023,
        event_month=1,
        max_post=8,
        label="Haifa privatization pre-war (2023-01; post<=8)",
    ),
    # Privatization extended: through 2024-12 => m=23
    ReformConfig(
        reform="haifa_priv_long",
        event_year=2023,
        event_month=1,
        max_post=23,
        label="Haifa privatization extended (2023-01; post<=23)",
    ),
]


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def find_thesis_root() -> Path:
    """
    Best-effort THESIS_ROOT detection.

    In your repo, scripts live in: THESIS_ROOT/Design/Code (new)/
    so THESIS_ROOT is two parents up from this file.
    """
    here = Path(__file__).resolve()
    # If user keeps a folder literally named "Thesis", prefer that.
    for p in [here] + list(here.parents):
        if p.name.lower() == "thesis":
            return p
    return here.parents[2]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normal_p_two_sided(t: float) -> float:
    """Two-sided p-value under N(0,1)."""
    if not np.isfinite(t):
        return np.nan
    z = abs(float(t))
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return 2.0 * (1.0 - cdf)


def select_cluster_series(df_kl: pd.DataFrame) -> pd.DataFrame:
    if "series_id" not in df_kl.columns:
        raise KeyError("KL panel missing required column: series_id")
    sub = df_kl[df_kl["series_id"] == CLUSTER_SERIES_ID].copy()
    if sub.empty:
        examples = df_kl["series_id"].dropna().drop_duplicates().head(10).tolist()
        raise ValueError(
            f"No rows for series_id={CLUSTER_SERIES_ID!r}. Example series_ids: {examples}"
        )
    return sub


def build_es_sample(df_cluster: pd.DataFrame, cfg: ReformConfig) -> pd.DataFrame:
    """Build a bounded event-time sample for one reform config."""
    df = df_cluster.copy()

    # Compute event_time
    df["event_time"] = (
        (df["year"].astype(int) - int(cfg.event_year)) * 12
        + (df["month"].astype(int) - int(cfg.event_month))
    )

    # Keep one extra pre month as a baseline bucket (m = -13) like other scripts
    lo = MIN_EVENT_TIME - 1
    hi = int(cfg.max_post)
    df = df[(df["event_time"] >= lo) & (df["event_time"] <= hi)].copy()

    # Common time index within the retained sample
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df["t_index"] = np.arange(len(df), dtype=int)

    return df


def add_window_dummies(df: pd.DataFrame, cfg: ReformConfig) -> pd.DataFrame:
    """Add pre/post coarse window dummies based on cfg.max_post."""
    df = df.copy()

    # Pre-trend window
    df["pre_all"] = ((df["event_time"] >= MIN_EVENT_TIME) & (df["event_time"] <= -2)).astype(int)

    # Post year 1
    y1_end = int(min(12, cfg.max_post))
    df["post_y1"] = ((df["event_time"] >= 1) & (df["event_time"] <= y1_end)).astype(int)

    # Post year 2 (only if any months)
    if cfg.max_post >= 13:
        df["post_y2"] = ((df["event_time"] >= 13) & (df["event_time"] <= int(cfg.max_post))).astype(int)
    else:
        df["post_y2"] = 0

    return df


def compute_post_all_from_parts(
    params: pd.Series,
    cov: pd.DataFrame,
    n_post_y1: int,
    n_post_y2: int,
) -> Tuple[float, float, float]:
    """Compute post_all as an observation-count-weighted avg of post_y1/post_y2."""
    n_total = int(n_post_y1) + int(n_post_y2)
    if n_total <= 0:
        return (np.nan, np.nan, np.nan)

    w1 = float(n_post_y1) / n_total
    w2 = float(n_post_y2) / n_total

    b1 = float(params.get("post_y1", np.nan))
    b2 = float(params.get("post_y2", 0.0)) if n_post_y2 > 0 else 0.0

    beta = w1 * b1 + w2 * b2

    # Var(w1*b1 + w2*b2)
    if n_post_y2 > 0 and ("post_y1" in cov.index) and ("post_y2" in cov.index):
        var = (
            (w1**2) * float(cov.loc["post_y1", "post_y1"]) +
            (w2**2) * float(cov.loc["post_y2", "post_y2"]) +
            2.0 * w1 * w2 * float(cov.loc["post_y1", "post_y2"])
        )
    elif "post_y1" in cov.index:
        var = (w1**2) * float(cov.loc["post_y1", "post_y1"])
    else:
        var = np.nan

    se = math.sqrt(var) if np.isfinite(var) and var >= 0 else np.nan
    t = beta / se if np.isfinite(beta) and np.isfinite(se) and se > 0 else np.nan
    return (beta, se, t)


# ---------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------

def estimate_for_cfg(df_es: pd.DataFrame, cfg: ReformConfig) -> Tuple[List[Dict], List[Dict]]:
    """Run TS window regressions for one cfg. Returns (window_rows, pretrend_rows)."""

    # Count months in each post part (for post_all weighting)
    n_post_y1 = int(df_es["post_y1"].sum())
    n_post_y2 = int(df_es["post_y2"].sum())

    window_rows: List[Dict] = []
    pretrend_rows: List[Dict] = []

    # Two simple specs: baseline and +trend
    specs = [
        ("ts_baseline", False),
        ("ts_trend", True),
    ]

    for spec_name, include_trend in specs:
        rhs_terms = ["pre_all", "post_y1"]
        if n_post_y2 > 0:
            rhs_terms.append("post_y2")
        if include_trend:
            rhs_terms.append("t_index")

        formula = "log_KL ~ " + " + ".join(rhs_terms)
        res = smf.ols(formula=formula, data=df_es).fit(cov_type="HC1")

        params = res.params
        cov = res.cov_params()

        # Pretrend test: H0 pre_all = 0
        try:
            ft = res.f_test("pre_all = 0")
            stat = float(ft.fvalue)
            pval = float(ft.pvalue)
            df_num = int(getattr(ft, "df_num", 1))
            df_denom = int(getattr(ft, "df_denom", res.df_resid))
        except Exception:
            stat = np.nan
            pval = np.nan
            df_num = np.nan
            df_denom = np.nan

        pretrend_rows.append(
            {
                "model": MODEL_NAME,
                "reform": cfg.reform,
                "target": "Haifa port cluster",
                "spec_name": spec_name,
                "test_name": "pre_all_eq_0",
                "stat": stat,
                "pvalue": pval,
                "df_num": df_num,
                "df_denom": df_denom,
                "cov_type": str(res.cov_type),
                "cluster_by": "",
                "r2": float(res.rsquared),
                "max_post": int(cfg.max_post),
            }
        )

        # --- pre_all ---
        b_pre = float(params.get("pre_all", np.nan))
        se_pre = float(res.bse.get("pre_all", np.nan))
        t_pre = b_pre / se_pre if np.isfinite(b_pre) and np.isfinite(se_pre) and se_pre > 0 else np.nan

        window_rows.append(
            {
                "model": MODEL_NAME,
                "reform": cfg.reform,
                "target": "Haifa port cluster",
                "spec_name": spec_name,
                "window_name": "pre_all",
                "m_start": MIN_EVENT_TIME,
                "m_end": -2,
                "n_months": int((-2) - MIN_EVENT_TIME + 1),
                "beta_hat": b_pre,
                "se": se_pre,
                "tvalue": t_pre,
                "pvalue": normal_p_two_sided(t_pre),
                "n_obs": int(res.nobs),
                "cov_type": str(res.cov_type),
                "cluster_by": "",
                "r2": float(res.rsquared),
                "max_post": int(cfg.max_post),
            }
        )

        # --- post_y1 ---
        y1_end = int(min(12, cfg.max_post))
        b_y1 = float(params.get("post_y1", np.nan))
        se_y1 = float(res.bse.get("post_y1", np.nan))
        t_y1 = b_y1 / se_y1 if np.isfinite(b_y1) and np.isfinite(se_y1) and se_y1 > 0 else np.nan

        window_rows.append(
            {
                "model": MODEL_NAME,
                "reform": cfg.reform,
                "target": "Haifa port cluster",
                "spec_name": spec_name,
                "window_name": "post_y1",
                "m_start": 1,
                "m_end": y1_end,
                "n_months": int(n_post_y1),
                "beta_hat": b_y1,
                "se": se_y1,
                "tvalue": t_y1,
                "pvalue": normal_p_two_sided(t_y1),
                "n_obs": int(res.nobs),
                "cov_type": str(res.cov_type),
                "cluster_by": "",
                "r2": float(res.rsquared),
                "max_post": int(cfg.max_post),
            }
        )

        # --- post_y2 ---
        if n_post_y2 > 0:
            b_y2 = float(params.get("post_y2", np.nan))
            se_y2 = float(res.bse.get("post_y2", np.nan))
            t_y2 = b_y2 / se_y2 if np.isfinite(b_y2) and np.isfinite(se_y2) and se_y2 > 0 else np.nan

            window_rows.append(
                {
                    "model": MODEL_NAME,
                    "reform": cfg.reform,
                    "target": "Haifa port cluster",
                    "spec_name": spec_name,
                    "window_name": "post_y2",
                    "m_start": 13,
                    "m_end": int(cfg.max_post),
                    "n_months": int(n_post_y2),
                    "beta_hat": b_y2,
                    "se": se_y2,
                    "tvalue": t_y2,
                    "pvalue": normal_p_two_sided(t_y2),
                    "n_obs": int(res.nobs),
                    "cov_type": str(res.cov_type),
                    "cluster_by": "",
                    "r2": float(res.rsquared),
                    "max_post": int(cfg.max_post),
                }
            )
        else:
            # placeholder (usually never used, because mediation won't request y2 if max_post<13)
            window_rows.append(
                {
                    "model": MODEL_NAME,
                    "reform": cfg.reform,
                    "target": "Haifa port cluster",
                    "spec_name": spec_name,
                    "window_name": "post_y2",
                    "m_start": 13,
                    "m_end": int(cfg.max_post),
                    "n_months": 0,
                    "beta_hat": np.nan,
                    "se": np.nan,
                    "tvalue": np.nan,
                    "pvalue": np.nan,
                    "n_obs": int(res.nobs),
                    "cov_type": str(res.cov_type),
                    "cluster_by": "",
                    "r2": float(res.rsquared),
                    "max_post": int(cfg.max_post),
                }
            )

        # --- post_all (computed) ---
        b_all, se_all, t_all = compute_post_all_from_parts(
            params, cov, n_post_y1=n_post_y1, n_post_y2=n_post_y2
        )
        window_rows.append(
            {
                "model": MODEL_NAME,
                "reform": cfg.reform,
                "target": "Haifa port cluster",
                "spec_name": spec_name,
                "window_name": "post_all",
                "m_start": 1,
                "m_end": int(cfg.max_post),
                "n_months": int(n_post_y1 + n_post_y2),
                "beta_hat": b_all,
                "se": se_all,
                "tvalue": t_all,
                "pvalue": normal_p_two_sided(t_all),
                "n_obs": int(res.nobs),
                "cov_type": str(res.cov_type),
                "cluster_by": "",
                "r2": float(res.rsquared),
                "max_post": int(cfg.max_post),
            }
        )

    return window_rows, pretrend_rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print(f"=== {MODEL_NAME}: starting ===")

    thesis_root = find_thesis_root()
    kl_path = thesis_root / "Data" / "KL" / "KL_Panel_monthly.tsv"
    out_dir = thesis_root / "Design" / "Output (new)" / "Model_1B_Haifa"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_win = out_dir / "model1b_haifa_window_betas.tsv"
    out_pre = out_dir / "model1b_haifa_pretrend_tests.tsv"

    print("THESIS_ROOT:", thesis_root)
    print("KL panel:", kl_path)
    print("Output window TSV:", out_win)
    print("Output pretrend TSV:", out_pre)

    df_kl = pd.read_csv(kl_path, sep="\t")

    required_cols = {"series_id", "year", "month", "log_KL"}
    missing = required_cols.difference(df_kl.columns)
    if missing:
        raise KeyError(f"KL panel missing required columns: {sorted(missing)}")

    df_cluster = select_cluster_series(df_kl)

    all_win_rows: List[Dict] = []
    all_pre_rows: List[Dict] = []

    for cfg in REFORMS:
        print(f"--- {cfg.label} | reform={cfg.reform} | max_post={cfg.max_post} ---")
        df_es = build_es_sample(df_cluster, cfg)
        df_es = add_window_dummies(df_es, cfg)

        if df_es.empty:
            print("  [WARN] Empty ES sample; skipping.")
            continue

        # Diagnostics
        lo = int(df_es["event_time"].min())
        hi = int(df_es["event_time"].max())
        print(f"  event_time support in-sample: [{lo}, {hi}] (n={len(df_es)})")

        win_rows, pre_rows = estimate_for_cfg(df_es, cfg)
        all_win_rows.extend(win_rows)
        all_pre_rows.extend(pre_rows)

    win_df = pd.DataFrame(all_win_rows)
    pre_df = pd.DataFrame(all_pre_rows)

    win_df.to_csv(out_win, sep="\t", index=False)
    pre_df.to_csv(out_pre, sep="\t", index=False)

    print(f"Wrote window rows: {len(win_df)} -> {out_win}")
    print(f"Wrote pretrend rows: {len(pre_df)} -> {out_pre}")
    print(f"=== {MODEL_NAME}: done ===")


if __name__ == "__main__":
    main()
