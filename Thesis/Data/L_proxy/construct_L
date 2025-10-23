#!/usr/bin/env python3
"""
Labor proxy (containers) — terminal×month and port×month, 2020–2024
-------------------------------------------------------------------
This single-file script rebuilds the L proxy with:
  • Complete monthly port TEU to 2024 using within-quarter filling that matches quarterly totals.
  • Terminal×month allocation based on observed terminal quarterly shares.
  • Annual hours from KPI TEU/hour, monthly disaggregation by terminal's own TEU shares.
  • Tolerant QA that understands partial quarters.

Inputs (TSV):
  Data/Output/teu_monthly_plus_quarterly_by_port.tsv
  Data/L_proxy/containers_kpis_annual_wide_filled.tsv

Outputs (TSV/JSON, in OUT_DIR):
  labor_hours_monthly_terminal.tsv
  labor_hours_monthly_port.tsv
  labor_hours_QA.tsv
  _meta_l_proxy.json

Usage (from repo root):
  python Data/L_proxy/build_labor_proxy_v2.py \
    --teu-panel Data/Output/teu_monthly_plus_quarterly_by_port.tsv \
    --kpi-wide  Data/L_proxy/containers_kpis_annual_wide_filled.tsv \
    --out-dir   Data/L_proxy \
    --start-year 2020 --end-year 2024 \
    --alloc equal --abs-tol 1000 --rel-tol 0.005 --reconcile-strict yes

Notes:
  • Default window is 2020–2024 because quarterly terminal TEU (CBS) is available 2020–2024.
  • Extend as your quarterly table extends.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ------------------------------
# Canonical names & mapping
# ------------------------------
TERMINAL_ALIAS_TO_CANON = {
    # legacy in quarterly table
    "Ashdod": "Ashdod-Legacy",
    "Haifa": "Haifa-Legacy",
    "Eilat": "Eilat",
    # new terminals
    "Ashdod HCT": "Ashdod-HCT",
    "HCT": "Ashdod-HCT",
    "נמל הדרום": "Ashdod-HCT",
    "Haifa SIPG": "Haifa-Bayport",
    "SIPG": "Haifa-Bayport",
    "נמל המפרץ": "Haifa-Bayport",
}

TERMINAL_TO_PORT = {
    "Ashdod-Legacy": "Ashdod",
    "Ashdod-HCT": "Ashdod",
    "Haifa-Legacy": "Haifa",
    "Haifa-Bayport": "Haifa",
    "Eilat": "Eilat",
}

PORTS_KEEP = {"Ashdod", "Haifa", "Eilat"}

# ------------------------------
# Helpers
# ------------------------------

def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def coalesce_teu(df: pd.DataFrame) -> pd.Series:
    # start with TEU if present
    if "TEU" in df.columns:
        out = pd.to_numeric(df["TEU"], errors="coerce")
    else:
        out = pd.Series(np.nan, index=df.index, dtype=float)
    # fill from TEU_thousands when TEU is missing
    if "TEU_thousands" in df.columns:
        out = out.fillna(pd.to_numeric(df["TEU_thousands"], errors="coerce") * 1000.0)
    if out.isna().all():
        raise ValueError("Could not find TEU or TEU_thousands in TEU panel.")
    return out.astype(float)



def parse_quarter(series: pd.Series) -> pd.Series:
    q = series.astype(str).str.extract(r"Q(\d)", expand=False)
    return pd.to_numeric(q, errors="coerce").astype("Int64")


def parse_month(series: pd.Series) -> pd.Series:
    m = series.astype(str).str.slice(0, 2)
    return pd.to_numeric(m, errors="coerce").astype("Int64")


def months_of_quarter(q: int) -> Tuple[int, int, int]:
    m1 = (q - 1) * 3 + 1
    return (m1, m1 + 1, m1 + 2)


def month_pos_in_quarter(m: int) -> int:
    return ((m - 1) % 3) + 1  # 1,2,3

# ------------------------------
# Load & normalize panel
# ------------------------------

def load_teu_panel(path: Path, start_year: int, end_year: int):
    df = pd.read_csv(path, sep="\t")
    df["TEU"] = coalesce_teu(df)

    # Monthly port TEU (observed)
    m = df.loc[df["Freq"].eq("Monthly")].copy()
    m = m.loc[m["Port"].isin(PORTS_KEEP)]
    m["year"] = pd.to_numeric(m["Year"], errors="coerce").astype("Int64")
    m["month"] = parse_month(m["Period"]).astype("Int64")
    m["quarter"] = (1 + (m["month"] - 1) // 3).astype("Int64")
    m = m.rename(columns={"Port": "port"})
    m = m.loc[(m["year"] >= start_year) & (m["year"] <= end_year)]
    m = (m[["port", "year", "month", "quarter", "TEU"]]
           .rename(columns={"TEU": "TEU_port_m_obs"})
           .sort_values(["port", "year", "month"]))

    # Quarterly terminal TEU (as provided, typically 2020–2024)
    q = df.loc[df["Freq"].eq("Quarterly")].copy()
    q["year"] = pd.to_numeric(q["Year"], errors="coerce").astype("Int64")
    q["quarter"] = parse_quarter(q["Period"]).astype("Int64")
    q["terminal"] = q["Port"].map(lambda s: TERMINAL_ALIAS_TO_CANON.get(s, s))
    q["port"] = q["terminal"].map(TERMINAL_TO_PORT)
    q = q[["port", "terminal", "year", "quarter", "TEU"]].rename(columns={"TEU": "TEU_i_q"})
    q = q.dropna(subset=["port", "terminal", "year", "quarter"]).astype({"year": int, "quarter": int})
    q = q.loc[(q["year"] >= start_year) & (q["year"] <= end_year)].copy()

    # --- Legacy backfill for years where we have monthly but no quarterly terminal rows ---
    # Port-quarter universe from monthly obs
    port_q_from_months = (m.groupby(["port", "year", "quarter"], as_index=False)["TEU_port_m_obs"]
                            .sum().rename(columns={"TEU_port_m_obs": "TEU_port_q_from_m"}))
    # Port-quarter from quarterly terminals
    port_q_from_term = (q.groupby(["port", "year", "quarter"], as_index=False)["TEU_i_q"]
                          .sum().rename(columns={"TEU_i_q": "TEU_port_q"}))

    # Union of quarters
    port_q_union = port_q_from_months.merge(
        port_q_from_term, on=["port", "year", "quarter"], how="outer"
    )

    # Where quarterly terminal data is missing but monthly exists → use monthly sum
    port_q_union["TEU_port_q"] = np.where(
        port_q_union["TEU_port_q"].notna(),
        port_q_union["TEU_port_q"],
        port_q_union["TEU_port_q_from_m"]
    )
    port_q_union = port_q_union.dropna(subset=["TEU_port_q"])
    port_q_union["year"] = port_q_union["year"].astype(int)
    port_q_union["quarter"] = port_q_union["quarter"].astype(int)

    # Build synthetic q_term rows for legacy-only quarters
    # Identify {port,year,quarter} not present in q
    key = ["port", "year", "quarter"]
    have_q = set(map(tuple, q[key].drop_duplicates().itertuples(index=False, name=None)))
    synth_rows = []
    for r in port_q_union.itertuples(index=False):
        k = (r.port, int(r.year), int(r.quarter))
        if k in have_q:
            continue
        # Only legacy terminal pre-2020 (no new entrants)
        if r.port == "Ashdod":
            terminal = "Ashdod-Legacy"
        elif r.port == "Haifa":
            terminal = "Haifa-Legacy"
        elif r.port == "Eilat":
            terminal = "Eilat"
        else:
            continue
        synth_rows.append({
            "port": r.port, "terminal": terminal,
            "year": int(r.year), "quarter": int(r.quarter),
            "TEU_i_q": float(r.TEU_port_q)
        })

    q_synth = pd.DataFrame(synth_rows)
    if not q_synth.empty:
        q = pd.concat([q, q_synth], ignore_index=True)

    # Final port_q from (possibly augmented) q
    port_q = (q.groupby(["port", "year", "quarter"], as_index=False)["TEU_i_q"].sum()
                .rename(columns={"TEU_i_q": "TEU_port_q"}))

    return m, q, port_q


# ------------------------------
# Seasonal weights (optional allocator)
# ------------------------------

def compute_seasonal_weights(m_obs: pd.DataFrame, ref_year_lo: int, ref_year_hi: int) -> pd.DataFrame:
    """Return per-port month-position weights within quarter (pos 1/2/3), normalized to 1.
    Uses observed monthly TEU in the reference window; if insufficient data, will yield NaNs.
    Columns: port, pos (1..3), w
    """
    df = m_obs.copy()
    df = df.loc[(df["year"] >= ref_year_lo) & (df["year"] <= ref_year_hi)].copy()
    if df.empty:
        return pd.DataFrame({"port": [], "pos": [], "w": []})
    df["pos"] = df["month"].astype(int).map(month_pos_in_quarter)
    g = df.groupby(["port", "year", "quarter"], as_index=False)["TEU_port_m_obs"].sum().rename(columns={"TEU_port_m_obs": "sum_q"})
    df = df.merge(g, on=["port", "year", "quarter"], how="left")
    df["w_m_in_q"] = np.where(df["sum_q"] > 0, df["TEU_port_m_obs"] / df["sum_q"], np.nan)
    w = (df.groupby(["port", "pos"], as_index=False)["w_m_in_q"].mean().rename(columns={"w_m_in_q": "w"}))
    # normalize per port so sum over pos 1..3 equals 1
    s = w.groupby("port", as_index=False)["w"].sum().rename(columns={"w": "wsum"})
    w = w.merge(s, on="port", how="left")
    w["w"] = np.where(w["wsum"] > 0, w["w"] / w["wsum"], np.nan)
    return w[["port", "pos", "w"]]

# ------------------------------
# Build complete monthly port TEU that matches quarterly totals
# ------------------------------

def build_port_month_full(m_obs: pd.DataFrame,
                          port_q: pd.DataFrame,
                          alloc: str = "equal",
                          seasonal_w: pd.DataFrame | None = None,
                          abs_tol: float = 1000.0,
                          rel_tol: float = 0.005,
                          reconcile_strict: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (M_port_full, coverage_df).
    M_port_full: port, year, quarter, month, TEU_port_m (observed or filled)
    coverage_df: per (port, year, quarter) — month_coverage_in_q (0..3), is_partial_quarter (bool)
    """
    # construct a skeleton of all months implied by port_q
    rows = []
    for _, r in port_q.iterrows():
        p, y, qtr, total_q = r["port"], int(r["year"]), int(r["quarter"]), float(r["TEU_port_q"])
        for m in months_of_quarter(qtr):
            rows.append({"port": p, "year": y, "quarter": qtr, "month": m, "TEU_port_m": np.nan, "TEU_port_q": total_q})
    full = pd.DataFrame(rows)

    # attach observed
    m2 = m_obs.rename(columns={"TEU_port_m_obs": "TEU_port_m_obs_"})
    full = full.merge(m2, on=["port", "year", "quarter", "month"], how="left")
    full["TEU_port_m"] = full["TEU_port_m_obs_"].astype(float)

    # coverage on raw obs
    cov = (full.groupby(["port", "year", "quarter"], as_index=False)["TEU_port_m_obs_"].apply(lambda s: s.notna().sum())
            .rename(columns={"TEU_port_m_obs_": "month_coverage_in_q"}))
    cov["is_partial_quarter"] = cov["month_coverage_in_q"] < 3

    # per quarter allocation
    def fill_one(group: pd.DataFrame) -> pd.DataFrame:
        total_q = float(group["TEU_port_q"].iloc[0])
        obs = group["TEU_port_m"].copy()
        obs_sum = float(np.nansum(obs.values)) if obs.notna().any() else 0.0
        n_obs = int(obs.notna().sum())
        n_miss = 3 - n_obs
        tol = max(abs_tol, rel_tol * max(total_q, 1.0))
        gap = total_q - obs_sum

        if n_miss == 0:
            if reconcile_strict and abs(gap) > tol:
                # rescale all three months proportionally
                if obs_sum > 0:
                    factor = total_q / obs_sum
                    group.loc[:, "TEU_port_m"] = obs * factor
                else:
                    # all zeros but total_q>0 — split equally
                    group.loc[:, "TEU_port_m"] = total_q / 3.0
            # else leave as-is
            return group

        # there are missing months
        if gap < -tol and reconcile_strict:
            # Observed months exceed quarter total beyond tolerance: rescale observed; missing -> 0
            if obs_sum > 0:
                factor = total_q / obs_sum
                group.loc[:, "TEU_port_m"] = obs.fillna(0.0) * factor
            else:
                group.loc[:, "TEU_port_m"] = 0.0
            return group

        if gap <= tol and gap < 0:  # small negative within tolerance → set missing to 0
            group.loc[group["TEU_port_m"].isna(), "TEU_port_m"] = 0.0
            return group

        # allocate positive gap to missing months
        miss_idx = group["TEU_port_m"].isna()
        if n_miss > 0:
            if alloc == "equal" or seasonal_w is None:
                add = np.full(n_miss, gap / n_miss if n_miss else 0.0)
            elif alloc == "seasonal":
                # use per-port month-position weights
                port = group["port"].iloc[0]
                pos = group["month"].map(month_pos_in_quarter)
                ww = seasonal_w.loc[seasonal_w["port"].eq(port)].set_index("pos")["w"].to_dict()
                miss_pos = pos[miss_idx].astype(int).tolist()
                raw = np.array([ww.get(p, np.nan) for p in miss_pos], dtype=float)
                if np.isnan(raw).all():
                    add = np.full(n_miss, gap / n_miss)
                else:
                    raw = np.where(np.isnan(raw), np.nanmean(raw), raw)
                    raw = np.maximum(raw, 1e-12)
                    add = gap * (raw / raw.sum())
            else:
                add = np.full(n_miss, gap / n_miss if n_miss else 0.0)
            group.loc[miss_idx, "TEU_port_m"] = add
        return group

    full = full.groupby(["port", "year", "quarter"], group_keys=False).apply(fill_one)
    full = full[["port", "year", "quarter", "month", "TEU_port_m"]].sort_values(["port", "year", "month"])  # final
    return full, cov

# ------------------------------
# Shares and allocation to terminals
# ------------------------------

def compute_quarter_shares(q: pd.DataFrame, port_q: pd.DataFrame) -> pd.DataFrame:
    s = q.merge(port_q, on=["port", "year", "quarter"], how="left")
    s["share_i_p_q"] = np.where(s["TEU_port_q"] > 0, s["TEU_i_q"] / s["TEU_port_q"], 0.0)
    s.loc[s["TEU_i_q"].eq(0) | s["TEU_i_q"].isna(), "share_i_p_q"] = 0.0
    # ensure sum of shares per (p,y,q) ≈ 1 when TEU_port_q>0
    return s[["port", "terminal", "year", "quarter", "TEU_i_q", "TEU_port_q", "share_i_p_q"]]


def allocate_terminal_monthly(m_port_full: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    mm = m_port_full.merge(shares, on=["port", "year", "quarter"], how="left")
    mm["share_i_p_q"] = mm["share_i_p_q"].fillna(0.0)
    mm["TEU_i_m"] = mm["share_i_p_q"] * mm["TEU_port_m"]
    cols = ["port", "terminal", "year", "quarter", "month", "TEU_port_m", "share_i_p_q", "TEU_i_m"]
    return mm[cols].sort_values(["port", "terminal", "year", "month"])  # terminal×month

# ------------------------------
# KPI & hours
# ------------------------------

def load_kpi(path: Path) -> pd.DataFrame:
    k = pd.read_csv(path, sep="\t")
    # be robust to casing
    cols = {c.lower(): c for c in k.columns}
    need = ["port", "terminal", "year", "teu_per_work_hour"]
    if not all(c in cols for c in need):
        raise ValueError("KPI wide table must contain: port, terminal, year, teu_per_work_hour")
    k = k[[cols["port"], cols["terminal"], cols["year"], cols["teu_per_work_hour"]]]
    k = k.rename(columns={cols["port"]: "port", cols["terminal"]: "terminal", cols["year"]: "year", cols["teu_per_work_hour"]: "Pi_teu_per_hour_i_y"})
    k = k.dropna(subset=["Pi_teu_per_hour_i_y"])  # restrict to years with published Pi
    k["year"] = k["year"].astype(int)
    return k


def compute_hours(teu_i_m: pd.DataFrame, kpi: pd.DataFrame) -> pd.DataFrame:
    ann = (teu_i_m.groupby(["port", "terminal", "year"], as_index=False)["TEU_i_m"].sum()
                  .rename(columns={"TEU_i_m": "TEU_i_y"}))
    ann = ann.merge(kpi, on=["port", "terminal", "year"], how="inner")  # keep only years with Pi
    ann["H_annual_i_y"] = ann["TEU_i_y"] / ann["Pi_teu_per_hour_i_y"]

    w = teu_i_m.merge(ann[["port", "terminal", "year", "TEU_i_y", "H_annual_i_y", "Pi_teu_per_hour_i_y"]],
                      on=["port", "terminal", "year"], how="inner")
    w["w_i_m_in_year"] = np.where(w["TEU_i_y"] > 0, w["TEU_i_m"] / w["TEU_i_y"], 0.0)
    w["L_hours_i_m"] = w["H_annual_i_y"] * w["w_i_m_in_year"]
    return w.sort_values(["port", "terminal", "year", "month"])

# ------------------------------
# QA
# ------------------------------

def build_qa(teu_i_m: pd.DataFrame,
             q: pd.DataFrame,
             port_cov: pd.DataFrame,
             abs_tol: float = 1000.0,
             rel_tol: float = 0.005) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # quarter reconciliation at terminal level
    qm = (teu_i_m.groupby(["port", "terminal", "year", "quarter"], as_index=False)["TEU_i_m"].sum()
                  .rename(columns={"TEU_i_m": "TEU_i_m_sum_q"}))
    chk = q.merge(qm, on=["port", "terminal", "year", "quarter"], how="left")
    chk["TEU_i_m_sum_q"].fillna(0.0, inplace=True)
    # attach port coverage to flag partial quarters
    chk = chk.merge(port_cov, on=["port", "year", "quarter"], how="left")
    chk["month_coverage_in_q"].fillna(0, inplace=True)
    chk["is_partial_quarter"] = chk["month_coverage_in_q"] < 3

    def ok_row(row):
        gap = abs(float(row["TEU_i_m_sum_q"]) - float(row["TEU_i_q"]))
        tol = max(abs_tol, rel_tol * max(float(row["TEU_i_q"]), 1.0))
        return gap <= tol

    chk["quarter_abs_gap_TEU"] = (chk["TEU_i_m_sum_q"] - chk["TEU_i_q"]).abs()
    chk["quarter_rel_gap_TEU"] = chk["quarter_abs_gap_TEU"] / chk["TEU_i_q"].replace({0: np.nan})

    quarter_ok = chk.apply(ok_row, axis=1)
    # set to NA where partial quarter (on raw obs)
    chk["quarter_sum_ok"] = np.where(chk["is_partial_quarter"], np.nan, quarter_ok)

    # preopening zero check: where TEU_i_q==0, all monthly should be 0
    zero_q = q[q["TEU_i_q"].eq(0)][["port", "terminal", "year", "quarter"]]
    z = (teu_i_m.merge(zero_q, on=["port", "terminal", "year", "quarter"], how="inner")
                 .assign(nonzero=lambda d: d["TEU_i_m"] > 0)
                 .groupby(["port", "terminal", "year", "quarter"], as_index=False)["nonzero"].any())
    z["preopen_zero_ok"] = ~z["nonzero"]

    qa_quarter = (chk[["port", "terminal", "year", "quarter", "month_coverage_in_q", "is_partial_quarter",
                       "quarter_abs_gap_TEU", "quarter_rel_gap_TEU", "quarter_sum_ok"]]
                    .merge(z[["port", "terminal", "year", "quarter", "preopen_zero_ok"]], how="left"))

    # annual reconciliation per terminal
    a_m = (teu_i_m.groupby(["port", "terminal", "year"], as_index=False)["TEU_i_m"].sum()
                   .rename(columns={"TEU_i_m": "TEU_i_y_from_m"}))
    a_q = (q.groupby(["port", "terminal", "year"], as_index=False)["TEU_i_q"].sum()
                 .rename(columns={"TEU_i_q": "TEU_i_y_from_q"}))
    qa_ann = a_q.merge(a_m, on=["port", "terminal", "year"], how="left").fillna({"TEU_i_y_from_m": 0.0})
    qa_ann["annual_abs_gap_TEU"] = (qa_ann["TEU_i_y_from_m"] - qa_ann["TEU_i_y_from_q"]).abs()
    qa_ann["annual_rel_gap_TEU"] = qa_ann["annual_abs_gap_TEU"] / qa_ann["TEU_i_y_from_q"].replace({0: np.nan})
    qa_ann["annual_sum_ok_TEU"] = qa_ann.apply(
        lambda r: r["annual_abs_gap_TEU"] <= max(abs_tol, rel_tol * max(r["TEU_i_y_from_q"], 1.0)), axis=1
    )

    return qa_ann, qa_quarter

# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build labor proxy (containers) terminal×month and port×month")
    ap.add_argument("--teu-panel", required=True)
    ap.add_argument("--kpi-wide", required=True)
    ap.add_argument("--out-dir", default="Data/L_proxy")
    ap.add_argument("--start-year", type=int, default=2020)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--alloc", choices=["equal", "seasonal"], default="equal")
    ap.add_argument("--seasonal-ref", default="2018-2021", help="YYYY-YYYY reference window for seasonal weights")
    ap.add_argument("--abs-tol", type=float, default=1000.0)
    ap.add_argument("--rel-tol", type=float, default=0.005)
    ap.add_argument("--reconcile-strict", choices=["yes", "no"], default="yes")
    args = ap.parse_args()

    teu_panel = Path(args.teu_panel)
    kpi_path = Path(args.kpi_wide)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    m_obs, q_term, port_q = load_teu_panel(teu_panel, args.start_year, args.end_year)

    # Seasonal weights if requested
    seasonal_w = None
    if args.alloc == "seasonal":
        lo, hi = map(int, args.seasonal_ref.split("-"))
        seasonal_w = compute_seasonal_weights(m_obs, lo, hi)

    # Build monthly port TEU complete to 2024
    m_port_full, port_cov = build_port_month_full(
        m_obs=m_obs,
        port_q=port_q,
        alloc=args.alloc,
        seasonal_w=seasonal_w,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
        reconcile_strict=(args.reconcile_strict == "yes"),
    )

    # Shares and allocation to terminal×month
    shares = compute_quarter_shares(q_term, port_q)
    teu_i_m = allocate_terminal_monthly(m_port_full, shares)

    # KPI and labor hours
    kpi = load_kpi(kpi_path)
    monthly_hours = compute_hours(teu_i_m, kpi)

    # QA
    qa_ann, qa_quarter = build_qa(teu_i_m, q_term, port_cov, abs_tol=args.abs_tol, rel_tol=args.rel_tol)

    # Write outputs
    terminal_out = out_dir / "labor_hours_monthly_terminal.tsv"
    monthly_hours[[
        "port", "terminal", "year", "month", "quarter",
        "TEU_port_m", "share_i_p_q", "TEU_i_m",
        "Pi_teu_per_hour_i_y", "H_annual_i_y", "w_i_m_in_year", "L_hours_i_m"
    ]].to_csv(terminal_out, sep="\t", index=False)

    port_out = out_dir / "labor_hours_monthly_port.tsv"
    port_m = (monthly_hours
              .groupby(["port", "year", "month", "quarter"], as_index=False)
              .agg(TEU_port_m=("TEU_port_m", "first"), L_hours_port_m=("L_hours_i_m", "sum")))
    port_m.to_csv(port_out, sep="\t", index=False)

    qa_out = out_dir / "labor_hours_QA.tsv"
    qa_quarter = qa_quarter.merge(qa_ann[["port", "terminal", "year", "annual_sum_ok_TEU"]],
                                  on=["port", "terminal", "year"], how="left")
    qa_quarter.to_csv(qa_out, sep="\t", index=False)

    # Meta
    meta = {
        "inputs": {
            "teu_panel": {"path": str(teu_panel), "sha256": sha256sum(teu_panel)},
            "kpi_table": {"path": str(kpi_path), "sha256": sha256sum(kpi_path)},
        },
        "options": {
            "start_year": args.start_year, "end_year": args.end_year,
            "alloc": args.alloc, "seasonal_ref": args.seasonal_ref,
            "abs_tol": args.abs_tol, "rel_tol": args.rel_tol,
            "reconcile_strict": args.reconcile_strict,
        },
        "rows": {
            "m_port_obs": len(m_obs), "q_term": len(q_term),
            "m_port_full": len(m_port_full), "terminal_months": len(monthly_hours)
        },
    }
    (out_dir / "_meta_l_proxy.json").write_text(json.dumps(meta, indent=2))

    print(f"✔ Wrote: {terminal_out}")
    print(f"✔ Wrote: {port_out}")
    print(f"✔ Wrote: {qa_out}")
    print(f"✔ Wrote: {out_dir / '_meta_l_proxy.json'}")


if __name__ == "__main__":
    main()



'''
python Data/L_proxy/construct_L \
  --teu-panel Data/Output/teu_monthly_plus_quarterly_by_port.tsv \
  --kpi-wide  Data/L_proxy/containers_kpis_annual_wide_filled.tsv \
  --out-dir   Data/L_proxy \
  --start-year 2018 --end-year 2024 \
  --alloc equal --abs-tol 1000 --rel-tol 0.005 --reconcile-strict yes
'''