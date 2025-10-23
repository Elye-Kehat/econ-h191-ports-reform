#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build LP panels with robust header handling and relaxed validation for L hours.

Key changes vs prior versions:
- Case-insensitive, whitespace-tolerant columns_map application.
- Fuzzy alias detection for critical columns (e.g., l_hours_i_m).
- Validation now WARNs (does not fail) if l_hours_i_m is missing; LP_id and L-based masks will be NA/looser.
- "month_index" is recomputed after merges; "quarter" normalized to 'Q1'..'Q4' as object before joins.

Inputs (TSV):
  1) L_Proxy.tsv  (terminal×month)
  2) monthly_output_by_1000_tons_ports_and_terminals.tsv
  3) teu_monthly_plus_quarterly_by_port.tsv

Optional:
  --columns_map path/to/columns_map.json  where each entry maps CANONICAL -> actual header in that file.
   Example for L_Proxy.tsv:
    {
      "L_Proxy.tsv": {
        "port": "port",
        "terminal": "terminal",
        "year": "year",
        "month": "month",
        "l_hours_i_m": "labor_hours",        # map if different
        "teu_i_m": "teu_i_m",
        "pi_teu_per_hour_i_y": "pi_teu_per_hour_i_y"
      }
    }

CLI (example):
  python build_lp_mixedfreq_final_v4.py \
    --l_proxy Data/LP/L_Proxy.tsv \
    --tons Data/LP/monthly_output_by_1000_tons_ports_and_terminals.tsv \
    --teu_mq Data/LP/teu_monthly_plus_quarterly_by_port.tsv \
    --columns_map Data/LP/columns_map.json \
    --out_dir Data/LP \
    --cutover "Haifa:2021-09,Ashdod:2022-07"

Outputs (TSV in out_dir):
  - LP_port_month_mixadjusted.tsv
  - LP_port_month_identity.tsv           (may be sparse if hours/teu monthly missing)
  - LP_terminal_month_mixadjusted.tsv
  - LP_terminal_quarter_mixadjusted.tsv
  - LP_panel_mixedfreq.tsv
  - qa_lp_report.tsv
  - _meta_lp_mixadjusted.json
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------- Errors & helpers --------------------------------

class ValidationError(Exception):
    pass

def _find_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _read_tsv_guess(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t", engine="python")
    except Exception as e:
        raise ValidationError(f"Failed to read TSV at {path}: {e}")

def _norm_port(s: str) -> str:
    if pd.isna(s):
        return s
    s2 = str(s).replace("–","-").strip()
    low = s2.lower()
    if low.startswith("ashdod"): return "Ashdod"
    if low.startswith("haifa"):  return "Haifa"
    if low.startswith("eilat"):  return "Eilat"
    if low in {"all ports","all_ports","allports","all"}: return "All Ports"
    return s2

def _quarter_from_month(m) -> Optional[str]:
    if pd.isna(m): return None
    try:
        q = (int(m)-1)//3 + 1
        return f"Q{q}"
    except Exception:
        return None

def _parse_quarter_field(q) -> Optional[int]:
    if pd.isna(q): return None
    s = str(q).upper().strip().replace(" ", "")
    m = re.search(r"Q([1-4])", s)
    if m: return int(m.group(1))
    if s.isdigit():
        qi = int(s)
        if 1 <= qi <= 4: return qi
    return None

def winsorize_group(df: pd.DataFrame, value_col: str, by: List[str], lower=0.01, upper=0.99) -> pd.Series:
    if df.empty: return df[value_col]
    out = pd.to_numeric(df[value_col], errors="coerce")
    g = df.groupby(by, dropna=False, sort=False)
    qs = g[value_col].quantile([lower, upper]).unstack(level=-1)
    if qs is None or qs.empty:
        return out
    qs = qs.rename(columns={lower:"q_low", upper:"q_high"})
    key = pd.MultiIndex.from_frame(df[by])
    ql = qs.reindex(key).reset_index(drop=True)["q_low"].to_numpy()
    qh = qs.reindex(key).reset_index(drop=True)["q_high"].to_numpy()
    v = out.to_numpy(dtype="float64")
    v = np.where(~np.isnan(v) & ~np.isnan(ql), np.maximum(v, ql), v)
    v = np.where(~np.isnan(v) & ~np.isnan(qh), np.minimum(v, qh), v)
    return pd.Series(v, index=df.index)

# ----------------------- Column mapping helpers ------------------------------

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    # strip whitespace and BOM, keep original cases in a dict, but we standardize for matching
    df = df.copy()
    df.columns = [str(c).replace("\ufeff","").strip() for c in df.columns]
    return df

def _apply_columns_map(df: pd.DataFrame, file_key: str, columns_map: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    columns_map uses key equal to basename (e.g., 'L_Proxy.tsv').
    It maps CANONICAL -> ACTUAL header in that file.
    We perform case-insensitive and whitespace-stripped matching of ACTUAL.
    """
    if not columns_map: return df
    bm = os.path.basename(file_key)
    mp = columns_map.get(bm) or columns_map.get(file_key) or {}
    if not mp: return df
    # Build rename dict by finding actual headers case-insensitively
    rename = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for canonical, actual in mp.items():
        actual_norm = str(actual).strip().lower()
        hit = lower_cols.get(actual_norm)
        if hit:
            rename[hit] = canonical
        else:
            # fallback: contains-based
            for c in df.columns:
                if actual_norm in c.lower():
                    rename[c] = canonical
                    break
    if rename:
        df = df.rename(columns=rename)
    return df

def _pick_col(df: pd.DataFrame, wanted: List[str]) -> Optional[str]:
    # exact (case-insensitive) then contains
    for cand in wanted:
        for c in df.columns:
            if c.lower() == cand.lower():
                return c
    for cand in wanted:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None

# ----------------------- Inputs & normalization ------------------------------

@dataclass
class Inputs:
    l_proxy_path: str
    tons_path: str
    teu_mq_path: str
    out_dir: str
    cutover: Dict[str, str]
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    columns_map_path: Optional[str] = None

def load_inputs(args) -> Inputs:
    base = args.base_dir
    l_proxy = _find_first_existing([
        args.l_proxy or "",
        os.path.join(base, "Data", "L_proxy", "L_Proxy.tsv"),
        os.path.join(base, "Data", "LP", "L_Proxy.tsv"),
        os.path.join(base, "L_Proxy.tsv"),
        os.path.join(base, "Data", "L_Proxy.tsv"),
    ])
    if not l_proxy:
        raise ValidationError("L_Proxy.tsv not found. Provide --l_proxy.")

    tons = _find_first_existing([
        args.tons or "",
        os.path.join(base, "Data", "Output", "monthly_output_by_1000_tons_ports_and_terminals.tsv"),
        os.path.join(base, "Data", "LP", "Output", "monthly_output_by_1000_tons_ports_and_terminals.tsv"),
        os.path.join(base, "monthly_output_by_1000_tons_ports_and_terminals.tsv"),
    ])
    if not tons:
        raise ValidationError("Tons file not found. Provide --tons.")

    teu_mq = _find_first_existing([
        args.teu_mq or "",
        os.path.join(base, "Data", "Output", "teu_monthly_plus_quarterly_by_port.tsv"),
        os.path.join(base, "Data", "LP", "Output", "teu_monthly_plus_quarterly_by_port.tsv"),
        os.path.join(base, "teu_monthly_plus_quarterly_by_port.tsv"),
    ])
    if not teu_mq:
        raise ValidationError("TEU file not found. Provide --teu_mq.")

    out_dir = args.out_dir or os.path.join(base, "Data", "LP")
    os.makedirs(out_dir, exist_ok=True)

    cut_map = {}
    if args.cutover:
        for kv in str(args.cutover).split(","):
            if ":" in kv:
                k, v = kv.split(":", 1)
                cut_map[k.strip()] = v.strip()
    if not cut_map:
        cut_map = {"Haifa": "2021-09", "Ashdod": "2022-07"}

    return Inputs(
        l_proxy_path=l_proxy,
        tons_path=tons,
        teu_mq_path=teu_mq,
        out_dir=out_dir,
        cutover=cut_map,
        winsor_lower=float(args.winsor_lower),
        winsor_upper=float(args.winsor_upper),
        columns_map_path=args.columns_map,
    )

# ----------------------- Loaders --------------------------------------------

def load_L_proxy(path: str, columns_map: Dict[str, Dict[str,str]]) -> pd.DataFrame:
    df = _normalize_headers(_read_tsv_guess(path))
    df = _apply_columns_map(df, os.path.basename(path), columns_map)

    # Detect columns with aliases
    port_col = _pick_col(df, ["port"])
    term_col = _pick_col(df, ["terminal"])
    year_col = _pick_col(df, ["year","yr"])
    month_col = _pick_col(df, ["month","mo"])

    l_hours_col = _pick_col(df, ["l_hours_i_m","l_hours","labor_hours","hours_i_m","hours"])
    teu_i_m_col = _pick_col(df, ["teu_i_m","teu_terminal","teu_m_terminal","teu"])
    pi_col      = _pick_col(df, ["pi_teu_per_hour_i_y","pi_i_y","pi","pi_teu_per_hour"])

    if port_col is None and term_col is None:
        raise ValidationError("L_Proxy: Could not find 'port' or 'terminal' columns. Check columns_map.json.")

    g = pd.DataFrame({
        "port": (df[port_col].astype(str).map(_norm_port) if port_col else pd.NA),
        "terminal": (df[term_col].astype(str).str.strip() if term_col else pd.NA),
        "year": pd.to_numeric(df[year_col], errors="coerce").astype("Int64"),
        "month": pd.to_numeric(df[month_col], errors="coerce").astype("Int64"),
        "l_hours_i_m": (pd.to_numeric(df[l_hours_col], errors="coerce") if l_hours_col else pd.NA),
        "teu_i_m": (pd.to_numeric(df[teu_i_m_col], errors="coerce") if teu_i_m_col else pd.NA),
        "pi_teu_per_hour_i_y": (pd.to_numeric(df[pi_col], errors="coerce") if pi_col else pd.NA),
    })
    g["quarter"] = g["month"].apply(_quarter_from_month)
    g["month_index"] = (g["year"].astype("float")*12 + g["month"].astype("float")).astype("Int64")
    return g

def load_tons_ports_and_terminals(path: str, columns_map: Dict[str, Dict[str,str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _normalize_headers(_read_tsv_guess(path))
    df = _apply_columns_map(df, os.path.basename(path), columns_map)

    # Two formats supported:
    # (A) Separate port, terminal, year, month, tons(_1000)
    # (B) A single "port_or_terminal" and "period" like "01-2008"
    if "port" in df.columns and "year" in df.columns and "month" in df.columns:
        port_col = "port"
        term_col = "terminal" if "terminal" in df.columns else None
        tons_col = "tons" if "tons" in df.columns else ("tons_1000" if "tons_1000" in df.columns else None)
        if tons_col is None:
            # try generic numeric
            tons_col = _pick_col(df, ["tons","tonnes","1000","value","amount","qty","quantity"])
        if tons_col is None:
            raise ValidationError("Tons file: could not find a numeric tons column.")
        tmp = pd.DataFrame({
            "port": df[port_col].astype(str).map(_norm_port),
            "terminal": (df[term_col].astype(str).str.strip() if term_col else pd.NA),
            "year": pd.to_numeric(df["year"], errors="coerce").astype("Int64"),
            "month": pd.to_numeric(df["month"], errors="coerce").astype("Int64"),
            "tons_raw": pd.to_numeric(df[tons_col], errors="coerce"),
        })
    elif "port_or_terminal" in df.columns and "period" in df.columns:
        # period like "01-2008" or "2008/01"
        s = df["period"].astype(str).str.strip()
        # try to parse
        ym = s.str.extract(r"(?P<m>\d{1,2})\D(?P<y>\d{4})")
        ym["y"] = pd.to_numeric(ym["y"], errors="coerce").astype("Int64")
        ym["m"] = pd.to_numeric(ym["m"], errors="coerce").astype("Int64")
        tmp = pd.DataFrame({
            "port": df["port_or_terminal"].astype(str).map(_norm_port),
            "terminal": pd.NA,
            "year": ym["y"],
            "month": ym["m"],
            "tons_raw": pd.to_numeric(df[_pick_col(df, ["tons","tons_1000","ktons","tons_k","value"])], errors="coerce"),
        })
    else:
        raise ValidationError("[Tons] Unrecognized schema. Provide columns_map.json to map headers.")

    # scale if in thousands
    name_lower = "tons"
    if "tons_1000" in df.columns: name_lower = "tons_1000"
    if "tons_k" in df.columns: name_lower = "tons_k"
    if ("1000" in name_lower) or ("k" in name_lower):
        tmp["tons"] = tmp["tons_raw"] * 1000.0
    else:
        tmp["tons"] = tmp["tons_raw"]

    is_all_ports = tmp["port"].astype(str).str.lower().isin(["all ports","all_ports","allports","all"])
    tons_all = tmp.loc[is_all_ports].copy()
    tons_port_term = tmp.loc[~is_all_ports].copy()

    is_port_total = tons_port_term["terminal"].isna() | (tons_port_term["terminal"].astype(str).str.strip()=="") | (tons_port_term["terminal"].str.lower().isin(["nan","none","na"]))
    tons_port = tons_port_term.loc[is_port_total].copy()
    tons_port["tons_source"] = "port_total"

    tons_term = tons_port_term.loc[~is_port_total].copy()
    tons_term_sum = tons_term.groupby(["port","year","month"], dropna=False)["tons"].sum(min_count=1).reset_index().rename(columns={"tons":"tons_sum_terminals"})

    tons_port_pref = tons_port[["port","year","month","tons","tons_source"]].rename(columns={"tons":"tons_p_m"})
    key = pd.concat([tons_port_pref[["port","year","month"]], tons_term_sum[["port","year","month"]]], ignore_index=True).drop_duplicates()
    merged = key.merge(tons_port_pref, on=["port","year","month"], how="left").merge(tons_term_sum, on=["port","year","month"], how="left")
    merged["tons_p_m"] = merged["tons_p_m"].combine_first(merged["tons_sum_terminals"])
    merged["tons_source"] = np.where(
        merged["tons_port_m"].notna() if "tons_port_m" in merged.columns else False, "port_total",
        np.where(merged["tons_sum_terminals"].notna(), "sum_terminals", "no_source")
    ) if "tons_port_m" in merged.columns else np.where(merged["tons_sum_terminals"].notna(), "sum_terminals", "port_total")
    # ensure object dtype
    merged["tons_source"] = merged["tons_source"].astype("object")

    tons_port_m = merged[["port","year","month","tons_p_m","tons_source"]].copy()
    tons_port_m["month_index"] = (tons_port_m["year"].astype("float")*12 + tons_port_m["month"].astype("float")).astype("Int64")

    tons_term_m = tons_term[["port","terminal","year","month","tons"]].rename(columns={"tons":"tons_i_m"}).copy()
    tons_allports_m = tons_all[["year","month","tons"]].rename(columns={"tons":"tons_allports_m"}).copy()
    return tons_port_m, tons_term_m, tons_allports_m

def load_teu_monthly_quarterly_by_port(path: str, columns_map: Dict[str, Dict[str,str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = _normalize_headers(_read_tsv_guess(path))
    df = _apply_columns_map(df, os.path.basename(path), columns_map)

    if "port" not in df.columns or "year" not in df.columns:
        raise ValidationError("TEU file needs at least port, year. Map via columns_map.json.")

    teu_m = pd.DataFrame(columns=["port","year","month","teu_p_m"])
    teu_q = pd.DataFrame(columns=["port","year","quarter","teu_p_q"])

    if "month" in df.columns:
        mpart = df[df["month"].notna()].copy()
        if not mpart.empty:
            teu_m = pd.DataFrame({
                "port": mpart["port"].astype(str).map(_norm_port),
                "year": pd.to_numeric(mpart["year"], errors="coerce").astype("Int64"),
                "month": pd.to_numeric(mpart["month"], errors="coerce").astype("Int64"),
                "teu_p_m": pd.to_numeric(mpart[_pick_col(mpart, ["teu","value","count","qty"])], errors="coerce")
            })
            teu_m["month_index"] = (teu_m["year"].astype("float")*12 + teu_m["month"].astype("float")).astype("Int64")

    if "quarter" in df.columns:
        qpart = df[df["quarter"].notna()].copy()
        if not qpart.empty:
            qnum = qpart["quarter"].apply(_parse_quarter_field)
            qpart["quarter"] = qnum.map({1:"Q1",2:"Q2",3:"Q3",4:"Q4"})
            teu_q = pd.DataFrame({
                "port": qpart["port"].astype(str).map(_norm_port),
                "year": pd.to_numeric(qpart["year"], errors="coerce").astype("Int64"),
                "quarter": qpart["quarter"].astype("object"),
                "teu_p_q": pd.to_numeric(qpart[_pick_col(qpart, ["teu","value","count","qty"])], errors="coerce")
            })

    return teu_m, teu_q

# ----------------------------- Validation ------------------------------------

def validate_inputs(l_proxy: pd.DataFrame, tons_port_m: pd.DataFrame, teu_pm: pd.DataFrame, teu_pq: pd.DataFrame) -> Tuple[bool, str]:
    msgs = []

    # L_Proxy checks (relaxed: hours optional)
    base_needed = ["port","terminal","year","month","teu_i_m","pi_teu_per_hour_i_y"]
    for col in base_needed:
        if col not in l_proxy.columns:
            msgs.append(f"[L_Proxy] Missing column: {col}")
    if "l_hours_i_m" not in l_proxy.columns:
        msgs.append("[L_Proxy] WARNING: 'l_hours_i_m' not found; LP_id will be NA and terminal masks will not use L.")

    # Month sanity
    if "month" in l_proxy.columns and l_proxy["month"].dropna().astype(int).between(1,12).eq(False).any():
        msgs.append("[L_Proxy] Found invalid month values outside 1..12.")

    dup_L = l_proxy.duplicated(["port","terminal","year","month"]).sum()
    if dup_L>0: msgs.append(f"[L_Proxy] Duplicate keys (port,terminal,year,month): {dup_L} rows.")

    # Tons checks
    for col in ["port","year","month","tons_p_m","tons_source"]:
        if col not in tons_port_m.columns: msgs.append(f"[Tons] Missing port-month column: {col}")
    dup_T = tons_port_m.duplicated(["port","year","month"]).sum()
    if dup_T>0: msgs.append(f"[Tons] Duplicate (port,year,month): {dup_T} rows.")

    # TEU presence
    ports = sorted(set(tons_port_m["port"].dropna().unique().tolist()))
    for p in ports:
        yrs = sorted(set(tons_port_m.loc[tons_port_m["port"]==p, "year"].dropna().unique().tolist()))
        for y in yrs:
            has_m = not teu_pm[(teu_pm["port"]==p) & (teu_pm["year"]==y)].empty
            has_q = not teu_pq[(teu_pq["port"]==p) & (teu_pq["year"]==y)].empty
            if not has_m and not has_q:
                msgs.append(f"[TEU] No monthly or quarterly TEU for port={p}, year={y}. w will be NA.")

    ok = (dup_L==0 and dup_T==0 and all(col in l_proxy.columns for col in base_needed))
    report = "\n".join(msgs) if msgs else "All validations passed."
    return ok, report

# ----------------------- Core computations ----------------------------------

def compute_w(tons_pm: pd.DataFrame, teu_pm: pd.DataFrame, teu_pq: pd.DataFrame,
              winsor_lower=0.01, winsor_upper=0.99) -> pd.DataFrame:
    # Monthly path
    w_m = tons_pm.merge(teu_pm, on=["port","year","month"], how="left")
    w_m["tons_per_teu"] = np.where(pd.to_numeric(w_m.get("teu_p_m"), errors="coerce")>0,
                                   pd.to_numeric(w_m.get("tons_p_m"), errors="coerce")/pd.to_numeric(w_m.get("teu_p_m"), errors="coerce"),
                                   np.nan)
    w_m["r_winsor"] = winsorize_group(w_m, "tons_per_teu", by=["port","year"], lower=winsor_lower, upper=winsor_upper)
    mean_by_py = w_m.groupby(["port","year"], dropna=False)["r_winsor"].transform("mean")
    w_m["w_p_m"] = np.where((mean_by_py==0) | (mean_by_py.isna()), 1.0, w_m["r_winsor"]/mean_by_py)
    w_m["w_p_m"] = w_m["w_p_m"].fillna(1.0)
    w_m["w_src_monthly"] = w_m["w_p_m"].apply(lambda x: "monthly" if pd.notna(x) else None).astype("object")
    w_m["month_index"] = (w_m["year"].astype("float")*12 + w_m["month"].astype("float")).astype("Int64")

    # Quarterly fallback
    if teu_pq.empty:
        w_qm = tons_pm[["port","year","month"]].copy()
        w_qm["w_from_q"] = np.nan
        w_qm["w_src_quarterly"] = np.nan
    else:
        tons_pq = tons_pm.copy()
        tons_pq["quarter"] = tons_pq["month"].apply(_quarter_from_month).astype("object")
        agg_tons = tons_pq.groupby(["port","year","quarter"], dropna=False)["tons_p_m"].sum(min_count=1).reset_index()
        rq = agg_tons.merge(teu_pq.assign(quarter=teu_pq["quarter"].astype("object")), on=["port","year","quarter"], how="left")
        rq["r_q"] = np.where(pd.to_numeric(rq.get("teu_p_q"), errors="coerce")>0,
                             pd.to_numeric(rq.get("tons_p_m"), errors="coerce")/pd.to_numeric(rq.get("teu_p_q"), errors="coerce"),
                             np.nan)
        rq["r_q_win"] = winsorize_group(rq, "r_q", by=["port","year"], lower=winsor_lower, upper=winsor_upper)
        mean_by_pyq = rq.groupby(["port","year"], dropna=False)["r_q_win"].transform("mean")
        rq["w_p_q"] = np.where((mean_by_pyq==0) | (mean_by_pyq.isna()), 1.0, rq["r_q_win"]/mean_by_pyq)

        map_q_to_m = tons_pm[["port","year","month"]].copy()
        map_q_to_m["quarter"] = map_q_to_m["month"].apply(_quarter_from_month).astype("object")
        w_qm = map_q_to_m.merge(rq[["port","year","quarter","w_p_q"]], on=["port","year","quarter"], how="left")
        w_qm = w_qm.rename(columns={"w_p_q":"w_from_q"})
        w_qm["w_src_quarterly"] = np.where(w_qm["w_from_q"].notna(), "quarterly", np.nan).astype("object")

    # Final w
    wf = w_m.merge(w_qm, on=["port","year","month"], how="outer", suffixes=("", "_q"))
    wf["month_index"] = (wf["year"].astype("float")*12 + wf["month"].astype("float")).astype("Int64")
    wf["w_final"] = wf["w_p_m"].combine_first(wf["w_from_q"])
    wf["w_source"] = wf["w_src_monthly"].combine_first(wf["w_src_quarterly"])
    wf["w_source"] = wf["w_source"].astype("object")
    return wf[["port","year","month","month_index","w_final","w_source"]]

def build_port_mix_LP(w_final: pd.DataFrame, l_proxy: pd.DataFrame, tons_pm: pd.DataFrame, teu_pm: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lp = l_proxy.copy()
    lp["quarter"] = lp["month"].apply(_quarter_from_month).astype("object")
    teui = (lp.groupby(["port","terminal","year","quarter"], dropna=False)["teu_i_m"]
              .sum(min_count=1).reset_index().rename(columns={"teu_i_m":"teu_i_q_sum"}))
    teutot = (teui.groupby(["port","year","quarter"], dropna=False)["teu_i_q_sum"]
                 .sum(min_count=1).reset_index().rename(columns={"teu_i_q_sum":"teu_port_q"}))
    shares = teui.merge(teutot, on=["port","year","quarter"], how="left")
    shares["share_i_q"] = np.where(pd.to_numeric(shares["teu_port_q"], errors="coerce")>0, shares["teu_i_q_sum"]/shares["teu_port_q"], np.nan)
    pi_i_y = (lp.groupby(["port","terminal","year"], dropna=False)["pi_teu_per_hour_i_y"].first().reset_index())
    shares = shares.merge(pi_i_y, on=["port","terminal","year"], how="left")
    pi_port_q = (shares.assign(pi_weighted=lambda d: d["share_i_q"]*d["pi_teu_per_hour_i_y"])
                      .groupby(["port","year","quarter"], dropna=False)["pi_weighted"]
                      .sum(min_count=1).reset_index().rename(columns={"pi_weighted":"Pi_p_q"}))
    months = w_final[["port","year","month","month_index"]].drop_duplicates()
    months["quarter"] = months["month"].apply(_quarter_from_month).astype("object")
    pi_pm = months.merge(pi_port_q, on=["port","year","quarter"], how="left")
    pi_pm = pi_pm.rename(columns={"Pi_p_q":"pi_p_y_mixbase"})

    lp_port = w_final.merge(pi_pm[["port","year","month","pi_p_y_mixbase"]], on=["port","year","month"], how="left")
    lp_port["lp_port_month_mix"] = pd.to_numeric(lp_port["w_final"], errors="coerce") * pd.to_numeric(lp_port["pi_p_y_mixbase"], errors="coerce")

    diag = tons_pm.merge(teu_pm, on=["port","year","month"], how="left")
    diag["month_index"] = (diag["year"].astype("float")*12 + diag["month"].astype("float")).astype("Int64")
    lp_port = lp_port.merge(diag[["port","year","month","month_index","tons_p_m","teu_p_m","tons_source"]],
                            on=["port","year","month","month_index"], how="left")

    L_port_m = (l_proxy.groupby(["port","year","month"], dropna=False)["l_hours_i_m"]
                        .sum(min_count=1).reset_index().rename(columns={"l_hours_i_m":"l_port_m"})) if "l_hours_i_m" in l_proxy.columns else pd.DataFrame(columns=["port","year","month","l_port_m"])
    lp_id = L_port_m.merge(teu_pm, on=["port","year","month"], how="left") if not L_port_m.empty else pd.DataFrame(columns=["port","year","month","lp_port_month_id"])
    if not lp_id.empty:
        lp_id["lp_port_month_id"] = np.where(pd.to_numeric(lp_id["l_port_m"], errors="coerce")>0,
                                             pd.to_numeric(lp_id["teu_p_m"], errors="coerce")/pd.to_numeric(lp_id["l_port_m"], errors="coerce"), np.nan)
        lp_id = lp_id[["port","year","month","lp_port_month_id"]]

    lp_port = lp_port.merge(L_port_m, on=["port","year","month"], how="left")
    lp_port = lp_port[["port","year","month","month_index","teu_p_m","tons_p_m","w_final","w_source",
                       "pi_p_y_mixbase","lp_port_month_mix","l_port_m","tons_source"]].copy()
    return lp_port, lp_id

def build_terminal_monthly(w_final: pd.DataFrame, l_proxy: pd.DataFrame) -> pd.DataFrame:
    df = l_proxy.merge(w_final[["port","year","month","w_final"]], on=["port","year","month"], how="left")
    df["lp_term_month_mixadjusted"] = pd.to_numeric(df["pi_teu_per_hour_i_y"], errors="coerce") * pd.to_numeric(df["w_final"], errors="coerce")
    if "l_hours_i_m" in df.columns:
        bad_L = (pd.to_numeric(df["l_hours_i_m"], errors="coerce")<=0)
    else:
        bad_L = pd.Series(False, index=df.index)
    bad_T = (pd.to_numeric(df["teu_i_m"], errors="coerce")<=0)
    df.loc[bad_L | bad_T, "lp_term_month_mixadjusted"] = np.nan

    out = df[["port","terminal","year","month","month_index","quarter",
              "pi_teu_per_hour_i_y","w_final","teu_i_m","l_hours_i_m","lp_term_month_mixadjusted"]].copy()
    return out

def aggregate_terminals_quarter_after_cutover(term_m: pd.DataFrame, cutover: Dict[str,str]) -> pd.DataFrame:
    cut_map: Dict[str,int] = {}
    for p, y_m in cutover.items():
        try:
            y, m = y_m.split("-")
            cut_map[p] = int(y)*12 + int(m)
        except Exception:
            cut_map[p] = 10**9

    term = term_m.copy()
    term["month_index"] = (term["year"].astype("int", errors="ignore")*12 + term["month"].astype("int", errors="ignore")).astype("Int64")
    term["quarter"] = term["month"].apply(_quarter_from_month).astype("object")
    term["freq"] = np.where(term["port"].map(cut_map).astype("Int64").le(term["month_index"]), "Q", "M")

    term_M = term[term["freq"]=="M"].copy()
    term_Q = term[term["freq"]=="Q"].copy()
    if not term_Q.empty:
        agg = term_Q.groupby(["port","terminal","year","quarter"], dropna=False).agg(
            pi_teu_per_hour_i_y=("pi_teu_per_hour_i_y","first"),
            w_final=("w_final","mean"),
            teu_i_m=("teu_i_m","sum"),
            l_hours_i_m=("l_hours_i_m","sum"),
            lp_term_month_mixadjusted=("lp_term_month_mixadjusted","mean"),
        ).reset_index()
        q_to_month = {"Q1":3,"Q2":6,"Q3":9,"Q4":12}
        agg["month"] = agg["quarter"].map(q_to_month).astype("Int64")
        agg["month_index"] = (agg["year"].astype("float")*12 + agg["month"].astype("float")).astype("Int64")
        agg["freq"] = "Q"
        term_Q_out = agg[["port","terminal","year","quarter","month","month_index","freq",
                          "pi_teu_per_hour_i_y","w_final","teu_i_m","l_hours_i_m","lp_term_month_mixadjusted"]]
    else:
        term_Q_out = pd.DataFrame(columns=["port","terminal","year","quarter","month","month_index","freq",
                          "pi_teu_per_hour_i_y","w_final","teu_i_m","l_hours_i_m","lp_term_month_mixadjusted"])

    term_M_out = term_M.copy()
    term_M_out["freq"] = "M"
    term_M_out = term_M_out[["port","terminal","year","quarter","month","month_index","freq",
                             "pi_teu_per_hour_i_y","w_final","teu_i_m","l_hours_i_m","lp_term_month_mixadjusted"]]

    out = pd.concat([term_M_out, term_Q_out], ignore_index=True).sort_values(["port","terminal","year","month"]).reset_index(drop=True)
    return out

def build_panel_mixedfreq(lp_port: pd.DataFrame, lp_id: pd.DataFrame, term_m: pd.DataFrame, term_qview: pd.DataFrame) -> pd.DataFrame:
    port = lp_port.copy()
    port["level"] = "port"; port["terminal"] = pd.NA
    port["Pi"] = port["pi_p_y_mixbase"]
    port["L_hours"] = port["l_port_m"]
    port["LP_mix"] = port["lp_port_month_mix"]
    if not lp_id.empty:
        port = port.merge(lp_id, on=["port","year","month"], how="left")
        port = port.rename(columns={"lp_port_month_id":"LP_id"})
    else:
        port["LP_id"] = pd.NA
    port["quarter"] = port["month"].apply(_quarter_from_month).astype("object")
    port["TEU"] = port["teu_p_m"]; port["tons"] = port["tons_p_m"]
    port["w"] = port["w_final"]; port["w_source"] = port["w_source"].astype("object")
    port["freq"] = "M"
    port_panel = port[["level","port","terminal","year","month","month_index","quarter","freq",
                       "TEU","tons","w","w_source","Pi","L_hours","LP_mix","LP_id","tons_source"]]

    term = term_qview.copy()
    term["level"] = "terminal"
    term = term.rename(columns={
        "pi_teu_per_hour_i_y":"Pi",
        "l_hours_i_m":"L_hours",
        "lp_term_month_mixadjusted":"LP_mix",
        "teu_i_m":"TEU",
        "w_final":"w",
    })
    term["LP_id"] = pd.NA
    term["tons"] = pd.NA
    term["w_source"] = pd.NA
    term["tons_source"] = pd.NA
    term_panel = term[["level","port","terminal","year","month","month_index","quarter","freq",
                       "TEU","tons","w","w_source","Pi","L_hours","LP_mix","LP_id","tons_source"]]

    panel = pd.concat([port_panel, term_panel], ignore_index=True).sort_values(["level","port","terminal","year","month"]).reset_index(drop=True)
    return panel

def run_qa(lp_port: pd.DataFrame, term_m: pd.DataFrame, w_final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def assert_unique(df, keys, name):
        c = df.duplicated(keys).sum()
        rows.append({"check":f"unique_keys_{name}", "result":"pass" if c==0 else "fail", "detail":f"duplicates={int(c)} keys={keys}"})
    assert_unique(lp_port, ["port","year","month"], "lp_port")
    assert_unique(term_m, ["port","terminal","year","month"], "lp_term_monthly")
    assert_unique(w_final, ["port","year","month"], "w_final")

    g = lp_port.groupby(["port","year"], dropna=False).agg(
        lp_mean=("lp_port_month_mix","mean"),
        pi_mean=("pi_p_y_mixbase","mean")
    ).reset_index()
    g["rel_err"] = np.abs(g["lp_mean"]-g["pi_mean"])/g["pi_mean"].replace(0,np.nan)
    for _, r in g.iterrows():
        rows.append({"check":"annual_preservation","port":r["port"],"year":int(r["year"]),
                     "lp_mean":float(r["lp_mean"]) if pd.notna(r["lp_mean"]) else None,
                     "pi_mean":float(r["pi_mean"]) if pd.notna(r["pi_mean"]) else None,
                     "rel_err":float(r["rel_err"]) if pd.notna(r["rel_err"]) else None,
                     "result":"pass" if (pd.isna(r["rel_err"]) or r["rel_err"]<=1e-6) else "warn"})
    src = w_final.assign(w_source=w_final["w_source"].astype("object")).groupby(["port","year","w_source"], dropna=False).size().reset_index(name="n")
    total = w_final.groupby(["port","year"], dropna=False).size().reset_index(name="N")
    src = src.merge(total, on=["port","year"], how="left")
    src["share"] = src["n"]/src["N"]
    for _, r in src.iterrows():
        rows.append({"check":"w_source_share","port":r["port"],"year":int(r["year"]),
                     "w_source":None if pd.isna(r["w_source"]) else str(r["w_source"]),
                     "n":int(r["n"]), "N":int(r["N"]), "share":float(r["share"])})
    return pd.DataFrame(rows)

# ------------------------------- Main ----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=".", help="Base directory of the repo; defaults to CWD.")
    ap.add_argument("--l_proxy", default=None, help="Path to L_Proxy.tsv (terminal×month).")
    ap.add_argument("--tons", default=None, help="Path to monthly_output_by_1000_tons_ports_and_terminals.tsv.")
    ap.add_argument("--teu_mq", default=None, help="Path to teu_monthly_plus_quarterly_by_port.tsv.")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: Data/LP under base_dir).")
    ap.add_argument("--columns_map", default=None, help="Path to columns_map.json (optional).")
    ap.add_argument("--cutover", default=None, help="Comma list like 'Haifa:2021-09,Ashdod:2022-07' (fallback).")
    ap.add_argument("--winsor_lower", type=float, default=0.01, help="Lower winsor quantile for r.")
    ap.add_argument("--winsor_upper", type=float, default=0.99, help="Upper winsor quantile for r.")
    args = ap.parse_args()

    try:
        inp = load_inputs(args)
        columns_map = {}
        if inp.columns_map_path and os.path.exists(inp.columns_map_path):
            with open(inp.columns_map_path, "r", encoding="utf-8") as f:
                columns_map = json.load(f)

        # Load inputs
        l_proxy = load_L_proxy(inp.l_proxy_path, columns_map)
        tons_port_m, tons_term_m, tons_allports_m = load_tons_ports_and_terminals(inp.tons_path, columns_map)
        teu_pm, teu_pq = load_teu_monthly_quarterly_by_port(inp.teu_mq_path, columns_map)

        # Validate (relaxed on hours)
        ok, report = validate_inputs(l_proxy, tons_port_m, teu_pm, teu_pq)
        print("VALIDATION REPORT:\n" + report)
        if not ok:
            sys.exit(1)

        # Compute w with quarterly fallback
        w_final = compute_w(
            tons_pm=tons_port_m,
            teu_pm=teu_pm,
            teu_pq=teu_pq,
            winsor_lower=inp.winsor_lower,
            winsor_upper=inp.winsor_upper,
        )

        # Build LP tables
        lp_port, lp_id = build_port_mix_LP(w_final, l_proxy, tons_port_m, teu_pm)
        term_m = build_terminal_monthly(w_final, l_proxy)
        term_qview = aggregate_terminals_quarter_after_cutover(term_m, inp.cutover)
        panel = build_panel_mixedfreq(lp_port, lp_id, term_m, term_qview)

        # QA
        qa = run_qa(lp_port, term_m, w_final)

        # Write outputs
        def _write_tsv(df: pd.DataFrame, name: str) -> str:
            path = os.path.join(inp.out_dir, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, sep="\t", index=False)
            return path

        _write_tsv(lp_port, "LP_port_month_mixadjusted.tsv")
        _write_tsv(lp_id, "LP_port_month_identity.tsv")
        _write_tsv(term_m, "LP_terminal_month_mixadjusted.tsv")
        _write_tsv(term_qview, "LP_terminal_quarter_mixadjusted.tsv")
        _write_tsv(panel, "LP_panel_mixedfreq.tsv")
        _write_tsv(qa, "qa_lp_report.tsv")

        meta = {
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "inputs": {"l_proxy": inp.l_proxy_path, "tons": inp.tons_path, "teu_mq": inp.teu_mq_path},
            "cutover": inp.cutover,
            "winsor": {"lower": inp.winsor_lower, "upper": inp.winsor_upper},
            "rows": {
                "LP_port_month_mixadjusted": int(len(lp_port)),
                "LP_port_month_identity": int(len(lp_id)),
                "LP_terminal_month_mixadjusted": int(len(term_m)),
                "LP_terminal_quarter_mixadjusted": int(len(term_qview)),
                "LP_panel_mixedfreq": int(len(panel)),
                "qa_lp_report": int(len(qa)),
            },
            "out_dir": inp.out_dir,
        }
        with open(os.path.join(inp.out_dir, "_meta_lp_mixadjusted.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print("Build completed. Outputs written to:", inp.out_dir)

    except ValidationError as ve:
        print(f"[VALIDATION ERROR] {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"[UNEXPECTED ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
