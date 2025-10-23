#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build monthly mix-adjusted LP by port and by terminal, with a unified long panel.

Key changes:
- Include Eilat in All-Ports allocation by default.
- Robust terminal name parsing (e.g., "Haifa SIPG", "Ashdod HCT", "Southport").
- Safer r = tons / TEU (no inf/NaN propagation; per-year mean-1 rebase with fallback).
- Coverage diagnostics by tons source in QA.
- Unified output table that stacks port and terminal rows into one tidy panel.

Defaults (repo-relative):
  TEU:       Data/Output/teu_monthly_plus_quarterly_by_port.tsv
             (falls back to Data/Output/teu_monthly_by_port.tsv if the first is missing)
  Tons:      Data/Output/monthly_output_by_1000_tons_ports_and_terminals.tsv
  L Proxy:   Data/L_proxy/L_Proxy.tsv
Outputs:
  Data/LP/LP_port_month_mixadjusted.tsv
  Data/LP/LP_terminal_month_mixadjusted.tsv
  Data/LP/LP_port_month_identity.tsv
  Data/LP/LP_panel.tsv                      <-- unified long table (port + terminal)
  Data/LP/qa_lp_report.tsv
  Data/LP/_meta_lp_mixadjusted.json
"""

import argparse
import json
import os
import hashlib
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

# -----------------------------
# Repo-relative defaults
# -----------------------------
HERE = Path(__file__).resolve()
DATA_DIR = HERE.parents[1]              # .../Thesis/Data
OUTPUT_DIR = DATA_DIR / "Output"
LPROXY_DIR = DATA_DIR / "L_proxy"
LP_DIR = DATA_DIR / "LP"

TEU_DEFAULT      = OUTPUT_DIR / "teu_monthly_plus_quarterly_by_port.tsv"
TEU_FALLBACK     = OUTPUT_DIR / "teu_monthly_by_port.tsv"
TONS_DEFAULT     = OUTPUT_DIR / "monthly_output_by_1000_tons_ports_and_terminals.tsv"
L_PROXY_DEFAULT  = LPROXY_DIR / "L_Proxy.tsv"

OUT_PORT_DEFAULT  = LP_DIR / "LP_port_month_mixadjusted.tsv"
OUT_TERM_DEFAULT  = LP_DIR / "LP_terminal_month_mixadjusted.tsv"
OUT_ID_DEFAULT    = LP_DIR / "LP_port_month_identity.tsv"
OUT_PANEL_DEFAULT = LP_DIR / "LP_panel.tsv"
OUT_QA_DEFAULT    = LP_DIR / "qa_lp_report.tsv"
OUT_META_DEFAULT  = LP_DIR / "_meta_lp_mixadjusted.json"

CANON_PORTS = ["Ashdod", "Haifa", "Eilat"]
# Per your instruction: include Eilat in the All-Ports allocation by default.
ALLOCATION_PORTS_DEFAULT = ["Ashdod", "Haifa", "Eilat"]

# -----------------------------
# Utils
# -----------------------------
def sha256sum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("-", "_", regex=True)
        .str.lower()
    )
    return df

def _norm_key(s: str) -> str:
    if pd.isna(s): return ""
    return " ".join(str(s).lower().strip().replace("-", " ").replace("_", " ").split())

def canon_port_name(x: str) -> str:
    s = _norm_key(x)
    if s in {"ashdod"}: return "Ashdod"
    if s in {"haifa"}: return "Haifa"
    if s in {"eilat"}: return "Eilat"
    if s in {"all ports", "allports", "all port"}: return "All Ports"
    # tolerant phrases
    if s.startswith("port of haifa"): return "Haifa"
    if s.startswith("port of ashdod"): return "Ashdod"
    if s.startswith("port of eilat"): return "Eilat"
    return str(x).strip() if not pd.isna(x) else x

def canon_terminal_name(x: str) -> str:
    """
    Map raw terminal labels to canonical terminals.
    Robust to spaces/hyphens/aliases like 'Haifa SIPG', 'Ashdod HCT', 'Southport', etc.
    """
    s = _norm_key(x)
    if not s: return np.nan

    # Ashdod side
    if ("ashdod" in s and "hct" in s) or ("southport" in s):
        return "Ashdod-HCT"
    if "ashdod" in s and "legacy" in s:
        return "Ashdod-Legacy"

    # Haifa side
    if ("haifa" in s) and ("sipg" in s or "bayport" in s):
        return "Haifa-Bayport"
    if "haifa" in s and "legacy" in s:
        return "Haifa-Legacy"

    # Exact canon names
    if s == _norm_key("Ashdod-HCT"): return "Ashdod-HCT"
    if s == _norm_key("Haifa-Bayport"): return "Haifa-Bayport"
    if s == _norm_key("Ashdod-Legacy"): return "Ashdod-Legacy"
    if s == _norm_key("Haifa-Legacy"): return "Haifa-Legacy"
    if s == _norm_key("Eilat"): return "Eilat"

    return np.nan  # leave as non-terminal row

def terminal_parent_port(term: str) -> str:
    t = str(term)
    if t.startswith("Ashdod-"): return "Ashdod"
    if t.startswith("Haifa-"): return "Haifa"
    if t == "Eilat": return "Eilat"
    return np.nan

def parse_month_year_to_ym(s: str) -> Tuple[int, int]:
    if pd.isna(s): return (np.nan, np.nan)
    s = str(s).strip()
    dt = None
    for fmt in [None, "%Y-%m", "%m-%Y", "%b-%Y", "%Y/%m", "%b %Y", "%Y %b"]:
        try:
            if fmt is None:
                dt = pd.to_datetime(s, errors="raise", infer_datetime_format=True)
            else:
                dt = pd.to_datetime(s, format=fmt, errors="raise")
            break
        except Exception:
            continue
    if dt is None or pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        raise ValueError(f"Could not parse Month-Year value: {s}")
    return int(dt.year), int(dt.month)

def write_tsv(df: pd.DataFrame, path: str, force: bool=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if (not force) and os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    df.to_csv(path, index=False, sep="\t")

# -----------------------------
# Loaders
# -----------------------------
def load_teu(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t|,", engine="python")
    df = normalize_columns(df)
    if "port" not in df.columns:
        raise ValueError("TEU file missing 'Port' column.")

    # Prefer explicit monthly in 'freq', with tolerant matching; else infer by presence of Month fields.
    if "freq" in df.columns and df["freq"].notna().any():
        freq_s = df["freq"].astype(str).str.lower()
        monthly_mask = freq_s.str.contains("month") | (freq_s.str.fullmatch(r"m"))
        dfm = df.loc[monthly_mask].copy()
        if dfm.empty:
            # fallback to all rows if 'freq' is unhelpful
            dfm = df.copy()
    else:
        dfm = df.copy()

    # Derive year & month if necessary
    if "year" not in dfm.columns:
        if "period" in dfm.columns:
            y, m = zip(*dfm["period"].map(parse_month_year_to_ym))
            dfm["year"], dfm["month"] = list(y), list(m)
        elif "monthindex" in dfm.columns:
            mi = dfm["monthindex"].astype(int)
            dfm["year"] = mi // 100
            dfm["month"] = mi % 100
        else:
            raise ValueError("TEU file lacks 'Year'/'Month' or parsable 'Period'/'MonthIndex'.")
    elif "month" not in dfm.columns and "monthindex" in dfm.columns:
        mi = dfm["monthindex"].astype(int)
        dfm["year"] = mi // 100
        dfm["month"] = mi % 100

    # TEU: prefer absolute TEU; otherwise convert from thousands
    if "teu" in dfm.columns:
        dfm["teu_p_m"] = pd.to_numeric(dfm["teu"], errors="coerce")
    elif "teu_thousands" in dfm.columns:
        dfm["teu_p_m"] = pd.to_numeric(dfm["teu_thousands"], errors="coerce") * 1000.0
    else:
        raise ValueError("TEU file must have 'TEU' or 'TEU_thousands'.")

    dfm["port"] = dfm["port"].map(canon_port_name)
    dfm = dfm.dropna(subset=["port", "year", "month"])
    dfm["year"] = dfm["year"].astype(int)
    dfm["month"] = dfm["month"].astype(int)

    keep_mask = dfm["port"].isin(CANON_PORTS + ["All Ports"])
    dfm = dfm.loc[keep_mask].copy()

    # Aggregate dupes
    dfm = dfm.groupby(["port", "year", "month"], as_index=False)["teu_p_m"].sum()

    dfm["month_index"] = dfm["year"] * 100 + dfm["month"]
    return dfm

def load_tons_mixed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t|,", engine="python")
    df = normalize_columns(df)
    required = ["portorterminal", "month_year", "tons_k"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Tons file missing columns: {missing}")

    y, m = zip(*df["month_year"].map(parse_month_year_to_ym))
    df["year"], df["month"] = list(y), list(m)

    # map labels
    df["portorterminal_raw"] = df["portorterminal"].astype(str)
    df["port_label"] = df["portorterminal_raw"].map(canon_port_name)
    df["terminal_label"] = df["portorterminal_raw"].map(canon_terminal_name)

    # convert tons_k to tons
    df["tons"] = pd.to_numeric(df["tons_k"], errors="coerce") * 1000.0
    return df

def load_l_proxy(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t|,", engine="python")
    df = normalize_columns(df)
    col_map = {}
    for c in df.columns:
        if c == "teu_i_m": col_map[c] = "teu_i_m"
        elif c in ("l_hours_i_m", "l_i_m", "l_hours"): col_map[c] = "l_hours_i_m"
        elif c in ("pi_teu_per_hour_i_y", "pi_i_y", "pi"): col_map[c] = "pi_teu_per_hour_i_y"
        elif c in ("quarter",): col_map[c] = "quarter"
        elif c in ("year",): col_map[c] = "year"
        elif c in ("month",): col_map[c] = "month"
        elif c in ("port",): col_map[c] = "port"
        elif c in ("terminal",): col_map[c] = "terminal"
    required = ["port", "terminal", "year", "month", "quarter", "teu_i_m", "l_hours_i_m", "pi_teu_per_hour_i_y"]
    for r in required:
        if r not in df.columns and r not in col_map.values():
            raise ValueError(f"L_Proxy missing required column: {r}")
    df = df.rename(columns=col_map)

    df["port"] = df["port"].map(canon_port_name)
    df["terminal"] = df["terminal"].map(lambda x: x if pd.isna(x) else str(x).strip())
    # Bring terminal names to canonical form if present
    df["terminal"] = df["terminal"].map(canon_terminal_name).fillna(df["terminal"])

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    if df["quarter"].dtype != object:
        df["quarter"] = df["quarter"].astype(str)
    df["quarter"] = df["quarter"].str.upper().str.replace(" ", "")
    df["month_index"] = df["year"] * 100 + df["month"]
    df["operating"] = np.where((pd.to_numeric(df["teu_i_m"], errors="coerce") > 0) &
                               (pd.to_numeric(df["l_hours_i_m"], errors="coerce") > 0), 1, 0)
    return df

# -----------------------------
# Core builders
# -----------------------------
def build_tons_pm(df_mixed: pd.DataFrame, df_teu_pm: pd.DataFrame, allocation_ports: List[str]) -> pd.DataFrame:
    # All Ports totals
    df_all = (
        df_mixed.loc[df_mixed["port_label"] == "All Ports", ["year", "month", "tons"]]
        .groupby(["year", "month"], as_index=False)["tons"].sum()
    )
    # Port totals
    df_port_rows = (
        df_mixed.loc[df_mixed["port_label"].isin(CANON_PORTS), ["port_label", "year", "month", "tons"]]
        .rename(columns={"port_label": "port"})
        .groupby(["port", "year", "month"], as_index=False)["tons"].sum()
    )
    # Terminal sums → parent port (using canonical terminal_label)
    df_term_rows = df_mixed.loc[df_mixed["terminal_label"].notna()].copy()
    df_term_rows["port"] = df_term_rows["terminal_label"].map(terminal_parent_port)
    df_term_rows = (
        df_term_rows.loc[df_term_rows["port"].isin(CANON_PORTS), ["port", "year", "month", "tons"]]
        .groupby(["port", "year", "month"], as_index=False)["tons"].sum()
    )

    base = df_teu_pm.loc[df_teu_pm["port"].isin(CANON_PORTS), ["port", "year", "month", "teu_p_m"]].copy()
    merged = (
        base
        .merge(df_port_rows, on=["port", "year", "month"], how="left")
        .rename(columns={"tons": "tons_porttotal"})
        .merge(df_term_rows, on=["port", "year", "month"], how="left")
        .rename(columns={"tons": "tons_terminalsum"})
        .merge(df_all, on=["year", "month"], how="left")  # 'tons' here is All Ports
    )

    # Allocation denominator over specified allocation ports (default includes Eilat now)
    alloc = (
        base.loc[base["port"].isin(allocation_ports)]
        .groupby(["year", "month"], as_index=False)["teu_p_m"].sum()
        .rename(columns={"teu_p_m": "teu_alloc_sum"})
    )
    merged = merged.merge(alloc, on=["year", "month"], how="left")

    # Decide tons precedence
    def decide_tons(row):
        if not pd.isna(row.get("tons_porttotal")):
            return row["tons_porttotal"], "port_total"
        if not pd.isna(row.get("tons_terminalsum")):
            return row["tons_terminalsum"], "sum_terminals"
        if (not pd.isna(row.get("tons"))) and (row["port"] in allocation_ports) and (row.get("teu_alloc_sum", 0) not in (0, np.nan)):
            share = (row["teu_p_m"] / row["teu_alloc_sum"]) if row["teu_alloc_sum"] else np.nan
            return row["tons"] * share, "allocated_allports"
        return np.nan, "no_source"

    tons_list, src_list = [], []
    for _, r in merged.iterrows():
        t, src = decide_tons(r)
        tons_list.append(t)
        src_list.append(src)

    merged["tons_p_m"]    = tons_list
    merged["tons_source"] = src_list
    merged["compare_diff"] = np.where(
        (~merged["tons_porttotal"].isna()) & (~merged["tons_terminalsum"].isna()),
        np.abs(merged["tons_porttotal"] - merged["tons_terminalsum"]) / (merged["tons_porttotal"].replace(0, np.nan)),
        np.nan
    )

    out = merged[["port", "year", "month", "teu_p_m", "tons_p_m", "tons_source", "compare_diff"]].copy()
    out["month_index"] = out["year"] * 100 + out["month"]
    return out

def winsorize_group(df: pd.DataFrame, value_col: str, by: List[str], lower=0.01, upper=0.99) -> pd.Series:
    def _clip(g):
        x = g[value_col].astype(float)
        finite = np.isfinite(x)
        if finite.sum() == 0:
            return x * np.nan
        lo = np.nanquantile(x[finite], lower)
        hi = np.nanquantile(x[finite], upper)
        y = x.copy()
        y[finite] = np.clip(x[finite], lo, hi)
        return y
    return df.groupby(by, group_keys=False).apply(_clip)

def compute_w(df_tons_pm: pd.DataFrame) -> pd.DataFrame:
    df = df_tons_pm.copy()
    # safer ratio: skip zero/neg TEU
    df["tons_per_teu"] = np.where(df["teu_p_m"] > 0, df["tons_p_m"] / df["teu_p_m"], np.nan)
    # winsorize within (port,year) on finite values
    df["r_winsor"] = winsorize_group(df, "tons_per_teu", by=["port", "year"], lower=0.01, upper=0.99)
    # rebase to mean 1 within (port,year); if group mean is NaN/0, set w=1
    mean_by_py = df.groupby(["port", "year"])["r_winsor"].transform("mean")
    df["w_p_m"] = np.where((mean_by_py==0) | (mean_by_py.isna()), 1.0, df["r_winsor"] / mean_by_py)
    df["w_p_m"] = df["w_p_m"].fillna(1.0)
    out = df[["port", "year", "month", "month_index", "teu_p_m", "tons_p_m", "tons_per_teu", "w_p_m", "tons_source"]].copy()
    return out

def compute_pi_mixbase_port_month(df_l_tm: pd.DataFrame) -> pd.DataFrame:
    # Quarterly terminal shares within port
    df_q = (
        df_l_tm.groupby(["port", "terminal", "year", "quarter"], as_index=False)["teu_i_m"]
        .sum()
        .rename(columns={"teu_i_m": "teu_i_q"})
    )
    df_q["sum_port_q"] = df_q.groupby(["port", "year", "quarter"])["teu_i_q"].transform("sum")
    df_q["s_i_p_q"] = np.where(df_q["sum_port_q"] > 0, df_q["teu_i_q"] / df_q["sum_port_q"], 0.0)

    df_m = df_l_tm[["port", "terminal", "year", "month", "quarter", "pi_teu_per_hour_i_y"]].copy()
    df_m = df_m.merge(
        df_q[["port", "terminal", "year", "quarter", "s_i_p_q"]],
        on=["port", "terminal", "year", "quarter"],
        how="left"
    )
    df_m["contrib"] = df_m["s_i_p_q"].fillna(0.0) * pd.to_numeric(df_m["pi_teu_per_hour_i_y"], errors="coerce")
    df_pi = (
        df_m.groupby(["port", "year", "month"], as_index=False)["contrib"]
        .sum()
        .rename(columns={"contrib": "pi_p_y_mixbase"})
    )
    df_pi["month_index"] = df_pi["year"] * 100 + df_pi["month"]
    return df_pi

def build_lp_port(df_w_pm: pd.DataFrame, df_pi_p_m: pd.DataFrame, df_l_tm: pd.DataFrame):
    # Primary port series
    df_lp = df_w_pm.merge(
        df_pi_p_m[["port", "year", "month", "pi_p_y_mixbase"]],
        on=["port", "year", "month"],
        how="left"
    )
    df_lp["lp_port_month_mix"] = df_lp["w_p_m"] * df_lp["pi_p_y_mixbase"]
    df_lp = df_lp[["port", "year", "month", "month_index", "teu_p_m", "tons_p_m", "tons_per_teu", "w_p_m", "pi_p_y_mixbase", "lp_port_month_mix", "tons_source"]].copy()

    # Identity (diagnostic) and L_port_m for unified panel
    df_L_port = (
        df_l_tm.groupby(["port", "year", "month"], as_index=False)["l_hours_i_m"]
        .sum()
        .rename(columns={"l_hours_i_m": "l_port_m"})
    )
    df_id = df_w_pm[["port","year","month","teu_p_m"]].merge(df_L_port, on=["port", "year", "month"], how="left")
    df_id["lp_port_month_id"] = np.where(df_id["l_port_m"]>0, df_id["teu_p_m"] / df_id["l_port_m"], np.nan)
    df_id = df_id[["port", "year", "month", "l_port_m", "lp_port_month_id"]].copy()

    df_lp_full = df_lp.merge(df_id, on=["port","year","month"], how="left")
    return df_lp_full, df_id

def build_lp_terminal(df_w_pm: pd.DataFrame, df_l_tm: pd.DataFrame) -> pd.DataFrame:
    df = df_l_tm.copy()
    df = df.merge(df_w_pm[["port", "year", "month", "w_p_m"]], on=["port", "year", "month"], how="left")
    df["lp_term_month_mixadjusted"] = pd.to_numeric(df["pi_teu_per_hour_i_y"], errors="coerce") * pd.to_numeric(df["w_p_m"], errors="coerce")
    # Structural zeros → keep NA
    df.loc[(pd.to_numeric(df["teu_i_m"], errors="coerce")<=0) | (pd.to_numeric(df["l_hours_i_m"], errors="coerce")<=0), "lp_term_month_mixadjusted"] = np.nan
    out = df[["port", "terminal", "year", "month", "month_index", "quarter", "operating", "pi_teu_per_hour_i_y", "w_p_m", "teu_i_m", "l_hours_i_m", "lp_term_month_mixadjusted"]].copy()
    return out

# -----------------------------
# QA + unified panel
# -----------------------------
def run_qa(df_lp_port: pd.DataFrame, df_lp_term: pd.DataFrame, df_w_pm: pd.DataFrame, df_pi_p_m: pd.DataFrame, df_tons_pm: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def assert_unique(df, keys, name):
        c = df.duplicated(keys).sum()
        rows.append({"check": f"unique_keys_{name}", "result": "pass" if c==0 else "fail", "detail": f"duplicates={c} keys={keys}"})
    assert_unique(df_lp_port, ["port","year","month"], "lp_port")
    assert_unique(df_lp_term, ["port","terminal","year","month"], "lp_term")
    assert_unique(df_w_pm, ["port","year","month"], "w_pm")
    assert_unique(df_pi_p_m, ["port","year","month"], "pi_p_m")
    assert_unique(df_tons_pm, ["port","year","month"], "tons_pm")

    # Annual preservation: mean LP_port == mean Pi_mixbase within (port,year)
    g = df_lp_port.groupby(["port","year"], as_index=False).agg(
        lp_mean=("lp_port_month_mix","mean"),
        pi_mean=("pi_p_y_mixbase","mean")
    )
    g["rel_err"] = np.abs(g["lp_mean"] - g["pi_mean"]) / g["pi_mean"].replace(0,np.nan)
    for _, r in g.iterrows():
        rows.append({
            "check": "annual_preservation",
            "port": r["port"],
            "year": int(r["year"]),
            "lp_mean": r["lp_mean"],
            "pi_mean": r["pi_mean"],
            "rel_err": r["rel_err"],
            "result": "pass" if (pd.isna(r["rel_err"]) or r["rel_err"]<=1e-6) else "warn"
        })

    # w variation (informational)
    cov = df_w_pm.groupby(["port","year"])["w_p_m"].agg(["mean","std"]).reset_index()
    cov["cv_w"] = cov["std"]/cov["mean"].replace(0,np.nan)
    for _, r in cov.iterrows():
        rows.append({"check":"w_variation_cv", "port":r["port"], "year":int(r["year"]), "cv_w": r["cv_w"], "result":"info"})

    # Coverage by tons_source
    src = df_tons_pm.groupby(["port","year","tons_source"]).size().unstack(fill_value=0)
    src = src.reset_index().rename_axis(None, axis=1)
    for _, rr in src.iterrows():
        total = int(rr.filter(regex="^(allocated_allports|port_total|sum_terminals|no_source)$").sum())
        rows.append({
            "check":"tons_source_breakdown",
            "port": rr["port"],
            "year": int(rr["year"]),
            "source_port_total": int(rr.get("port_total",0)),
            "source_sum_terminals": int(rr.get("sum_terminals",0)),
            "source_allocated_allports": int(rr.get("allocated_allports",0)),
            "source_no_source": int(rr.get("no_source",0)),
            "total_months": total,
            "result": "fail" if int(rr.get("no_source",0))>0 else "info"
        })

    return pd.DataFrame(rows)

def build_unified_panel(df_lp_port_full: pd.DataFrame, df_lp_term: pd.DataFrame) -> pd.DataFrame:
    # Port-level rows
    port = df_lp_port_full.copy()
    port["level"] = "port"
    port["terminal"] = pd.NA
    port["Pi"] = port["pi_p_y_mixbase"]
    port["L_hours"] = port["l_port_m"]
    port["LP_mix"] = port["lp_port_month_mix"]
    port["LP_id"] = port["lp_port_month_id"]
    # derive quarter
    port["quarter"] = "Q" + (((port["month"] - 1) // 3) + 1).astype(str)
    port["TEU"] = port["teu_p_m"]
    port["tons"] = port["tons_p_m"]
    port["tons_source"] = port["tons_source"]
    port["w"] = port["w_p_m"]
    port["tons_per_teu"] = port["tons_per_teu"]

    port_panel = port[[
        "level","port","terminal","year","month","month_index","quarter",
        "TEU","tons","tons_per_teu","w","Pi","L_hours","LP_mix","LP_id","tons_source"
    ]].copy()

    # Terminal-level rows
    term = df_lp_term.copy()
    term["level"] = "terminal"
    term["Pi"] = term["pi_teu_per_hour_i_y"]
    term["L_hours"] = term["l_hours_i_m"]
    term["LP_mix"] = term["lp_term_month_mixadjusted"]
    term["LP_id"] = pd.NA  # optional: could compute TEU_i_m / L_i_m as a diagnostic
    term["TEU"] = term["teu_i_m"]
    term["tons"] = pd.NA
    term["tons_per_teu"] = pd.NA
    term["w"] = term["w_p_m"]
    term["tons_source"] = pd.NA

    term_panel = term[[
        "level","port","terminal","year","month","month_index","quarter",
        "TEU","tons","tons_per_teu","w","Pi","L_hours","LP_mix","LP_id","tons_source"
    ]].copy()

    # Stack
    panel = pd.concat([port_panel, term_panel], ignore_index=True)
    # Sort for readability
    panel = panel.sort_values(["level","port","terminal","year","month"], kind="mergesort").reset_index(drop=True)
    return panel

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teu", default=str(TEU_DEFAULT))
    parser.add_argument("--tons", default=str(TONS_DEFAULT))
    parser.add_argument("--l-proxy", dest="l_proxy", default=str(L_PROXY_DEFAULT))
    parser.add_argument("--out-port", default=str(OUT_PORT_DEFAULT))
    parser.add_argument("--out-term", default=str(OUT_TERM_DEFAULT))
    parser.add_argument("--out-id", default=str(OUT_ID_DEFAULT))
    parser.add_argument("--out-panel", default=str(OUT_PANEL_DEFAULT))
    parser.add_argument("--out-qa", default=str(OUT_QA_DEFAULT))
    parser.add_argument("--out-meta", default=str(OUT_META_DEFAULT))
    parser.add_argument("--force", action="store_true")
    # Per your request, default now includes Eilat
    parser.add_argument("--allocation-ports", nargs="*", default=ALLOCATION_PORTS_DEFAULT)
    args = parser.parse_args()

    # Friendly fallback for TEU file
    if not os.path.exists(args.teu):
        alt = str(TEU_FALLBACK)
        if os.path.exists(alt):
            print(f"[info] TEU file not found at {args.teu}; using fallback {alt}")
            args.teu = alt

    # Load inputs
    teu        = load_teu(args.teu)
    tons_mixed = load_tons_mixed(args.tons)
    l_tm       = load_l_proxy(args.l_proxy)

    # Build tons (precedence incl. All-Ports allocation)
    tons_pm = build_tons_pm(tons_mixed, teu, allocation_ports=args.allocation_ports)

    # Compute w_{p,m}
    w_pm = compute_w(tons_pm)

    # Compute Pi_p_y mix-base at month grain
    pi_p_m = compute_pi_mixbase_port_month(l_tm)

    # Build LP (port) + identity (+ l_port_m)
    lp_port_full, lp_id = build_lp_port(w_pm, pi_p_m, l_tm)

    # Build LP (terminal)
    lp_term = build_lp_terminal(w_pm, l_tm)

    # QA
    qa = run_qa(lp_port_full, lp_term, w_pm, pi_p_m, tons_pm)

    # Unified panel
    panel = build_unified_panel(lp_port_full, lp_term)

    # META
    meta = {
        "script": str(HERE),
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "inputs": {
            "teu":   {"path": args.teu, "sha256": sha256sum(args.teu) if os.path.exists(args.teu) else None},
            "tons":  {"path": args.tons, "sha256": sha256sum(args.tons) if os.path.exists(args.tons) else None},
            "l_proxy":{"path": args.l_proxy, "sha256": sha256sum(args.l_proxy) if os.path.exists(args.l_proxy) else None},
        },
        "parameters": {
            "winsor_pct": [0.01, 0.99],
            "allocation_ports": args.allocation_ports,
            "force": bool(args.force),
        },
        "row_counts": {
            "teu_rows": int(len(teu)),
            "tons_mixed_rows": int(len(tons_mixed)),
            "l_proxy_rows": int(len(l_tm)),
            "tons_pm_rows": int(len(tons_pm)),
            "w_pm_rows": int(len(w_pm)),
            "lp_port_rows": int(len(lp_port_full)),
            "lp_term_rows": int(len(lp_term)),
            "panel_rows": int(len(panel)),
            "qa_rows": int(len(qa)),
        }
    }

    # Write
    write_tsv(lp_port_full.sort_values(["port","year","month"]), args.out_port, force=args.force)
    write_tsv(lp_term.sort_values(["port","terminal","year","month"]), args.out_term, force=args.force)
    write_tsv(lp_id.sort_values(["port","year","month"]), args.out_id, force=args.force)
    write_tsv(panel, args.out_panel, force=args.force)
    write_tsv(qa, args.out_qa, force=args.force)
    Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:")
    for p in [args.out_port, args.out_term, args.out_id, args.out_panel, args.out_qa, args.out_meta]:
        print("  ", p)

if __name__ == "__main__":
    main()
