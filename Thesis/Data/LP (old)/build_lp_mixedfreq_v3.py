#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_lp_mixedfreq_v3.py  —  Robust LP panel builder (error-proof, column-map aware)

Implements the agreed logic with mixed-frequency w:
- Monthly TEU -> monthly w where available
- Quarterly TEU -> quarterly w fallback, broadcast to months (w_source='quarterly')

Granularity policy
------------------
- Ports: monthly series (always). LP_mix uses w_final (monthly or quarterly-broadcast).
- Terminals: monthly source-of-truth; plus a Q-view after per-port cutover (freq='Q').

Inputs (TSV)
------------
  1) L_Proxy.tsv   (terminal×month; requires: port, terminal, year, month, l_hours_i_m, teu_i_m, pi_teu_per_hour_i_y)
  2) monthly_output_by_1000_tons_ports_and_terminals.tsv  (tons; may have PortOrTerminal + Month-Year + tons_k)
  3) teu_monthly_plus_quarterly_by_port.tsv (port-level TEU monthly and/or quarterly)

Optional config
---------------
  - Data/LP/columns_map.json  # maps canonical -> actual headers per file
  - --cutover "Haifa:2021-09,Ashdod:2022-07"
  - --allocate_allports (False by default; see notes below)

Outputs (TSV)
-------------
  - LP_port_month_mixadjusted.tsv
  - LP_port_month_identity.tsv
  - LP_terminal_month_mixadjusted.tsv
  - LP_terminal_quarter_mixadjusted.tsv
  - LP_panel_mixedfreq.tsv
  - qa_lp_report.tsv
  - _meta_lp_mixadjusted.json

Notes
-----
- We DO NOT synthesize monthly TEU from quarterly.
- w_source is carried through ('monthly' | 'quarterly').
- Terminal pre-opening months are NA for LP_mix (masking via L and TEU > 0).
- Annual preservation diag printed.
- All string/nullable safety handled to avoid NAType/int casting errors.
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

pd.options.mode.copy_on_write = True

# --------------------------- Errors & helpers --------------------------------

class ValidationError(Exception):
    pass

def _find_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _read_tsv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t", engine="python")
    except Exception as e:
        raise ValidationError(f"Failed to read TSV at {path}: {e}")

def _norm_port(s) -> Optional[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s2 = str(s).replace("–","-").strip()
    low = s2.lower()
    if low.startswith("ashdod"): return "Ashdod"
    if low.startswith("haifa"):  return "Haifa"
    if low.startswith("eilat"):  return "Eilat"
    if low in {"all ports","all_ports","allports","all"}: return "All Ports"
    # Try to infer from terminal composite name "Haifa-Legacy", etc.
    m = re.match(r"^(Ashdod|Haifa|Eilat)\b", s2, flags=re.IGNORECASE)
    if m:
        return m.group(1).title()
    return s2

def _parse_period_to_year_month(s) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse common month-year strings to (year, month). Supports:
    - '01-2008', '2008-01', '2008/01', '01/2008'
    - 'Mar-2022', '2022-Mar', 'Mar 2022', '2022 Mar'
    """
    if pd.isna(s):
        return (None, None)
    txt = str(s).strip()
    # Numeric month-year patterns
    m = re.match(r"^\s*(\d{1,2})[-/](\d{4})\s*$", txt)  # 01-2008
    if m:
        mo, yr = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12: return (yr, mo)
    m = re.match(r"^\s*(\d{4})[-/](\d{1,2})\s*$", txt)  # 2008-01
    if m:
        yr, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12: return (yr, mo)
    # Named month patterns
    months = {m.lower(): i for i, m in enumerate(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)}
    m = re.match(r"^\s*([A-Za-z]{3,})[-\s](\d{4})\s*$", txt)  # Mar 2022 / Mar-2022
    if m:
        mon = m.group(1)[:3].title()
        yr = int(m.group(2))
        mo = months.get(mon.lower())
        if mo: return (yr, mo)
    m = re.match(r"^\s*(\d{4})[-\s]([A-Za-z]{3,})\s*$", txt)  # 2022 Mar / 2022-Mar
    if m:
        yr = int(m.group(1))
        mon = m.group(2)[:3].title()
        mo = months.get(mon.lower())
        if mo: return (yr, mo)
    return (None, None)

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
    out = df[value_col].astype(float).copy()
    if out.empty: return out
    g = df.groupby(by, dropna=False, sort=False)
    qs = g[value_col].quantile([lower, upper]).unstack(level=-1)
    if qs is None or qs.empty: return out
    qs = qs.rename(columns={lower:"q_low", upper:"q_high"})
    key = pd.MultiIndex.from_frame(df[by])
    ql = qs.reindex(key).reset_index(drop=True)["q_low"].to_numpy()
    qh = qs.reindex(key).reset_index(drop=True)["q_high"].to_numpy()
    v = out.to_numpy(dtype="float64")
    v = np.where(~np.isnan(v) & ~np.isnan(ql), np.maximum(v, ql), v)
    v = np.where(~np.isnan(v) & ~np.isnan(qh), np.minimum(v, qh), v)
    return pd.Series(v, index=df.index)

# ---------------------------- Column map -------------------------------------

def _read_columns_map(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValidationError("columns_map.json must be a JSON object.")
            return data
    except Exception as e:
        raise ValidationError(f"Failed to read columns_map.json: {e}")

def _apply_header_map(df: pd.DataFrame, file_basename: str, columns_map: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    bm = os.path.basename(file_basename)
    mp = columns_map.get(bm) or columns_map.get(file_basename) or {}
    if not mp:
        return df
    rename_dict = {}
    for canonical, actual in mp.items():
        # if exact exists
        if actual in df.columns:
            rename_dict[actual] = canonical
        else:
            # try case-insensitive equal
            for c in df.columns:
                if c.lower() == str(actual).lower():
                    rename_dict[c] = canonical
                    break
    if rename_dict:
        df = df.rename(columns=rename_dict)
    return df

# ----------------------- Inputs & normalization ------------------------------

@dataclass
class Inputs:
    l_proxy_path: str
    tons_path: str
    teu_mq_path: str
    out_dir: str
    cutover: Dict[str, str]
    winsor_lower: float
    winsor_upper: float
    columns_map_path: Optional[str]
    allocate_allports: bool
    min_port_year_coverage: float

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
        raise ValidationError("monthly_output_by_1000_tons_ports_and_terminals.tsv not found. Provide --tons.")

    teu_mq = _find_first_existing([
        args.teu_mq or "",
        os.path.join(base, "Data", "Output", "teu_monthly_plus_quarterly_by_port.tsv"),
        os.path.join(base, "Data", "LP", "Output", "teu_monthly_plus_quarterly_by_port.tsv"),
        os.path.join(base, "teu_monthly_plus_quarterly_by_port.tsv"),
    ])
    if not teu_mq:
        raise ValidationError("teu_monthly_plus_quarterly_by_port.tsv not found. Provide --teu_mq.")

    out_dir = args.out_dir or os.path.join(base, "Data", "LP")
    os.makedirs(out_dir, exist_ok=True)

    # Cutover mapping
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
        allocate_allports=bool(args.allocate_allports),
        min_port_year_coverage=float(args.min_port_year_coverage),
    )

# ----------------------- Loaders --------------------------------------------

def load_L_proxy(path: str, columns_map: Dict[str, Dict[str,str]]) -> pd.DataFrame:
    df = _read_tsv(path)
    df = _apply_header_map(df, os.path.basename(path), columns_map)

    required = ["port","terminal","year","month","l_hours_i_m","teu_i_m","pi_teu_per_hour_i_y"]
    for r in required:
        if r not in df.columns:
            raise ValidationError(f"L_Proxy: missing '{r}'. Use columns_map.json to map headers.")

    g = pd.DataFrame({
        "port": df["port"].astype(str).map(_norm_port),
        "terminal": df["terminal"].astype(str).str.strip(),
        "year": pd.to_numeric(df["year"], errors="coerce").astype("Int64"),
        "month": pd.to_numeric(df["month"], errors="coerce").astype("Int64"),
        "l_hours_i_m": pd.to_numeric(df["l_hours_i_m"], errors="coerce"),
        "teu_i_m": pd.to_numeric(df["teu_i_m"], errors="coerce"),
        "pi_teu_per_hour_i_y": pd.to_numeric(df["pi_teu_per_hour_i_y"], errors="coerce"),
    })
    g["quarter"] = g["month"].apply(_quarter_from_month)
    g["month_index"] = (g["year"].astype("float")*12 + g["month"].astype("float")).round().astype("Int64")
    return g

def load_tons(path: str, columns_map: Dict[str, Dict[str,str]], allocate_allports: bool,
              teu_pm_for_alloc: Optional[pd.DataFrame], l_proxy_for_alloc: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Expect columns mapped to canonical:
    - port_or_terminal  (combined label)
    - period            (e.g., '01-2008')
    - tons_1000 or tons
    """
    df = _read_tsv(path)
    df = _apply_header_map(df, os.path.basename(path), columns_map)

    col_pot = "port_or_terminal" if "port_or_terminal" in df.columns else None
    col_period = "period" if "period" in df.columns else None
    col_tons_k = "tons_1000" if "tons_1000" in df.columns else None
    col_tons = "tons" if "tons" in df.columns else None

    if not col_pot or not col_period or (not col_tons_k and not col_tons):
        # Provide useful diagnostics
        first_cols = list(df.columns)[:10]
        sample = df.head(5).to_dict(orient="records")
        raise ValidationError(f"[monthly_output_by_1000_tons_ports_and_terminals.tsv] "
                              f"Could not parse required columns. First 10 columns: {first_cols}. "
                              f"Sample rows: {sample}. Tip: provide columns_map.json or rename headers.")

    # Parse year/month
    yy, mm = zip(*[ _parse_period_to_year_month(x) for x in df[col_period].tolist() ])
    tmp = pd.DataFrame({
        "raw_label": df[col_pot].astype(str).str.strip(),
        "year": pd.Series(yy, dtype="Int64"),
        "month": pd.Series(mm, dtype="Int64"),
        "tons_raw": pd.to_numeric(df[col_tons_k], errors="coerce")*1000.0 if col_tons_k else pd.to_numeric(df[col_tons], errors="coerce"),
    })
    if tmp["year"].isna().mean() > 0.99 or tmp["month"].isna().mean() > 0.99:
        first_cols = list(df.columns)[:10]
        sample = df.head(5).to_dict(orient="records")
        raise ValidationError(f"[monthly_output_by_1000_tons_ports_and_terminals.tsv] Could not parse year/month (>100% NA). "
                              f"First 10 columns: {first_cols}. Sample rows: {sample}. "
                              f"Ensure 'period' strings like '01-2008' or map via columns_map.json.")

    # Split raw_label into (port total / terminal / all-ports)
    lab = tmp["raw_label"].fillna("")
    is_all = lab.str.lower().isin(["all ports","all_ports","allports","all"])
    # Terminal if contains a hyphen after a port prefix; else if equals a port name -> port total
    is_terminal = lab.str.match(r"^(Ashdod|Haifa|Eilat)\s*[-–]\s*.+", flags=re.IGNORECASE)
    is_port_total = ~is_all & ~is_terminal

    # Build port and terminal fields
    tmp["port"] = None
    tmp.loc[is_port_total, "port"] = lab[is_port_total].map(_norm_port)
    tmp.loc[is_terminal, "port"] = lab[is_terminal].str.replace("–","-").str.extract(r"^(Ashdod|Haifa|Eilat)", flags=re.IGNORECASE)[0].str.title()
    tmp["terminal"] = None
    tmp.loc[is_terminal, "terminal"] = lab[is_terminal].str.replace("–","-").str.extract(r"^(Ashdod|Haifa|Eilat)\s*[-–]\s*(.+)$", flags=re.IGNORECASE)[1].str.strip()

    tmp["month_index"] = (tmp["year"].astype("float")*12 + tmp["month"].astype("float")).round().astype("Int64")

    # Separate frames
    tons_all = tmp.loc[is_all, ["year","month","month_index","tons_raw"]].rename(columns={"tons_raw":"tons_allports_m"}).copy()
    tons_term = tmp.loc[is_terminal, ["port","terminal","year","month","month_index","tons_raw"]].rename(columns={"tons_raw":"tons_i_m"}).copy()
    tons_port_tot = tmp.loc[is_port_total, ["port","year","month","month_index","tons_raw"]].rename(columns={"tons_raw":"tons"}).copy()
    tons_port_tot["tons_source"] = "port_total"

    # Sum terminal rows to port-month
    tons_term_sum = (tons_term.groupby(["port","year","month","month_index"], dropna=False)["tons_i_m"]
                     .sum(min_count=1).reset_index().rename(columns={"tons_i_m":"tons_sum_terminals"}))

    # Merge precedence
    key = pd.concat([tons_port_tot[["port","year","month","month_index"]], tons_term_sum[["port","year","month","month_index"]]], ignore_index=True).drop_duplicates()
    merged = key.merge(tons_port_tot, on=["port","year","month","month_index"], how="left").merge(tons_term_sum, on=["port","year","month","month_index"], how="left")
    merged["tons_p_m"] = merged["tons"].combine_first(merged["tons_sum_terminals"])

    # Optional: allocate All-Ports to ports when both missing (rare). Off by default.
    if allocate_allports:
        # Build port shares by month from TEU monthly if present else from terminal TEU_i_m aggregation
        if teu_pm_for_alloc is not None and not teu_pm_for_alloc.empty:
            shares = (teu_pm_for_alloc.groupby(["year","month"], dropna=False)
                      .apply(lambda d: d.set_index("port")["teu_p_m"]/d["teu_p_m"].sum())
                      .reset_index().rename(columns={0:"share"}))
        elif l_proxy_for_alloc is not None and not l_proxy_for_alloc.empty:
            teui = (l_proxy_for_alloc.groupby(["port","year","month"], dropna=False)["teu_i_m"].sum(min_count=1).reset_index().rename(columns={"teu_i_m":"teu_port_m"}))
            shares = (teui.groupby(["year","month"], dropna=False)
                      .apply(lambda d: d.set_index("port")["teu_port_m"]/d["teu_port_m"].sum())
                      .reset_index().rename(columns={0:"share"}))
        else:
            shares = pd.DataFrame(columns=["year","month","port","share"])

        # Fill missing tons_p_m using All-Ports * share
        alloc = merged.merge(tons_all, on=["year","month","month_index"], how="left") \
                      .merge(shares, on=["year","month","port"], how="left")
        need_alloc = alloc["tons_p_m"].isna() & alloc["tons_allports_m"].notna() & alloc["share"].notna()
        alloc.loc[need_alloc, "tons_p_m"] = alloc.loc[need_alloc, "tons_allports_m"] * alloc.loc[need_alloc, "share"]
        alloc["tons_source"] = np.where(merged["tons"].notna(), "port_total",
                                 np.where(merged["tons_sum_terminals"].notna(), "sum_terminals",
                                          np.where(need_alloc, "allocated_allports", "no_source")))
        tons_port_m = alloc[["port","year","month","month_index","tons_p_m","tons_source"]].copy()
    else:
        merged["tons_source"] = np.where(merged["tons"].notna(), "port_total",
                                  np.where(merged["tons_sum_terminals"].notna(), "sum_terminals","no_source"))
        tons_port_m = merged[["port","year","month","month_index","tons_p_m","tons_source"]].copy()

    return tons_port_m, tons_term, tons_all

def load_teu(path: str, columns_map: Dict[str, Dict[str,str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = _read_tsv(path)
    df = _apply_header_map(df, os.path.basename(path), columns_map)

    # Expect at least: port, year, (month or quarter), teu
    req_any = ["port","year","teu"]
    for r in req_any:
        if r not in df.columns:
            raise ValidationError(f"TEU file: missing '{r}'. Map with columns_map.json.")

    dfc = df.copy()
    dfc["port"] = dfc["port"].astype(str).map(_norm_port)
    dfc["year"] = pd.to_numeric(dfc["year"], errors="coerce").astype("Int64")
    dfc["teu"]  = pd.to_numeric(dfc["teu"], errors="coerce")

    # Monthly slice
    teu_m = pd.DataFrame(columns=["port","year","month","month_index","teu_p_m"])
    if "month" in dfc.columns:
        mpart = dfc[dfc["month"].notna()].copy()
        if not mpart.empty:
            mpart["month"] = pd.to_numeric(mpart["month"], errors="coerce").astype("Int64")
            teu_m = mpart[["port","year","month","teu"]].rename(columns={"teu":"teu_p_m"})
            teu_m["month_index"] = (teu_m["year"].astype("float")*12 + teu_m["month"].astype("float")).round().astype("Int64")

    # Quarterly slice
    teu_q = pd.DataFrame(columns=["port","year","quarter","teu_p_q"])
    if "quarter" in dfc.columns:
        qpart = dfc[dfc["quarter"].notna()].copy()
        if not qpart.empty:
            qnum = qpart["quarter"].apply(_parse_quarter_field)
            teu_q = qpart.assign(quarter=qnum.map({1:"Q1",2:"Q2",3:"Q3",4:"Q4"}))[["port","year","quarter","teu"]].rename(columns={"teu":"teu_p_q"})

    # If neither present, try 'period'
    if teu_m.empty and "period" in dfc.columns:
        per = dfc[dfc["period"].notna()].copy()
        yy, mm = zip(*[ _parse_period_to_year_month(x) for x in per["period"].tolist() ])
        per["year"] = pd.Series(yy, dtype="Int64").combine_first(per["year"])
        per["month"] = pd.Series(mm, dtype="Int64")
        mpart = per[per["month"].notna()].copy()
        if not mpart.empty:
            teu_m = mpart.assign(teu_p_m=mpart["teu"])[["port","year","month","teu_p_m"]]
            teu_m["month_index"] = (teu_m["year"].astype("float")*12 + teu_m["month"].astype("float")).round().astype("Int64")

    return teu_m, teu_q

# ----------------------- Validation & diagnostics ----------------------------

def validate_inputs(l_proxy: pd.DataFrame, tons_port_m: pd.DataFrame, teu_pm: pd.DataFrame, teu_pq: pd.DataFrame) -> Tuple[bool, str]:
    msgs = []

    # L_Proxy checks
    for col in ["port","terminal","year","month","l_hours_i_m","teu_i_m","pi_teu_per_hour_i_y"]:
        if col not in l_proxy.columns:
            msgs.append(f"[L_Proxy] Missing column: {col}")
    if l_proxy["month"].dropna().astype(int).between(1,12).eq(False).any():
        msgs.append(f"[L_Proxy] Invalid month values outside 1..12.")
    if l_proxy.duplicated(["port","terminal","year","month"]).sum() > 0:
        msgs.append(f"[L_Proxy] Duplicate keys (port,terminal,year,month).")

    # Tons checks
    for col in ["port","year","month","tons_p_m","tons_source"]:
        if col not in tons_port_m.columns:
            msgs.append(f"[Tons] Missing column in port-month tons: {col}")
    if tons_port_m.duplicated(["port","year","month"]).sum() > 0:
        msgs.append(f"[Tons] Duplicate (port,year,month).")

    # TEU presence (at least monthly or quarterly per port-year existing in tons)
    ports = sorted(set(tons_port_m["port"].dropna().unique().tolist()))
    for p in ports:
        yrs = sorted(set(tons_port_m.loc[tons_port_m["port"]==p, "year"].dropna().unique().tolist()))
        for y in yrs:
            has_m = not teu_pm[(teu_pm["port"]==p) & (teu_pm["year"]==y)].empty
            has_q = not teu_pq[(teu_pq["port"]==p) & (teu_pq["year"]==y)].empty
            if not has_m and not has_q:
                msgs.append(f"[TEU] No monthly or quarterly TEU for port={p}, year={y}. w will be NA for those months.")

    ok = (len([m for m in msgs if m.startswith("[L_Proxy] Missing") or m.startswith("[Tons] Missing")])==0)
    report = "\n".join(msgs) if msgs else "All validations passed."
    return ok, report

def crosswalk_ports(*dfs) -> None:
    print("—— Port name crosswalk (unique values per source) ——")
    for i, d in enumerate(dfs, start=1):
        if d is None or d.empty:
            print(f"[{i}] <empty>")
            continue
        col = "port" if "port" in d.columns else None
        if not col:
            print(f"[{i}] <no-port-col>")
            continue
        vals = sorted([str(x) for x in pd.Series(d[col].unique()).dropna().tolist()])
        print(f"[{i}] {vals[:40]}{' ...' if len(vals)>40 else ''}")

# ----------------------- Core computations ----------------------------------

def compute_w(tons_pm: pd.DataFrame, teu_pm: pd.DataFrame, teu_pq: pd.DataFrame,
              winsor_lower=0.01, winsor_upper=0.99) -> pd.DataFrame:
    # Monthly path
    w_m = tons_pm.merge(teu_pm, on=["port","year","month","month_index"], how="left")
    w_m["tons_per_teu"] = np.where(w_m["teu_p_m"]>0, w_m["tons_p_m"]/w_m["teu_p_m"], np.nan)
    if not w_m.empty:
        w_m["r_winsor"] = winsorize_group(w_m, "tons_per_teu", by=["port","year"], lower=winsor_lower, upper=winsor_upper)
        mean_by_py = w_m.groupby(["port","year"], dropna=False)["r_winsor"].transform("mean")
        w_m["w_p_m"] = np.where((mean_by_py==0) | (mean_by_py.isna()), np.nan, w_m["r_winsor"]/mean_by_py)
    else:
        w_m["w_p_m"] = np.nan
    w_m["w_src_monthly"] = np.where(w_m["w_p_m"].notna(), "monthly", None).astype("object")

    # Quarterly fallback
    # Prepare mapper from months -> quarter
    map_q = tons_pm[["port","year","month","month_index"]].drop_duplicates().copy()
    map_q["quarter"] = map_q["month"].apply(_quarter_from_month)

    if teu_pq is not None and not teu_pq.empty:
        agg = tons_pm.copy()
        agg["quarter"] = agg["month"].apply(_quarter_from_month)
        agg_tons = agg.groupby(["port","year","quarter"], dropna=False)["tons_p_m"].sum(min_count=1).reset_index()
        rq = agg_tons.merge(teu_pq, on=["port","year","quarter"], how="left")
        rq["r_q"] = np.where(rq["teu_p_q"]>0, rq["tons_p_m"]/rq["teu_p_q"], np.nan)
        rq["r_q_win"] = winsorize_group(rq, "r_q", by=["port","year"], lower=winsor_lower, upper=winsor_upper)
        mean_by_pyq = rq.groupby(["port","year"], dropna=False)["r_q_win"].transform("mean")
        rq["w_p_q"] = np.where((mean_by_pyq==0) | (mean_by_pyq.isna()), np.nan, rq["r_q_win"]/mean_by_pyq)
        w_qm = map_q.merge(rq[["port","year","quarter","w_p_q"]], on=["port","year","quarter"], how="left")
        w_qm = w_qm.rename(columns={"w_p_q":"w_from_q"})
        w_qm["w_src_quarterly"] = np.where(w_qm["w_from_q"].notna(), "quarterly", None).astype("object")
    else:
        w_qm = map_q.copy()
        w_qm["w_from_q"] = np.nan
        w_qm["w_src_quarterly"] = None

    # Final
    wf = w_m[["port","year","month","month_index","w_p_m","w_src_monthly"]].merge(
         w_qm[["port","year","month","month_index","w_from_q","w_src_quarterly"]],
         on=["port","year","month","month_index"], how="outer")
    wf["w_final"] = wf["w_p_m"].combine_first(wf["w_from_q"])
    wf["w_source"] = wf["w_src_monthly"].combine_first(wf["w_src_quarterly"]).astype("object")
    return wf[["port","year","month","month_index","w_final","w_source"]]

def build_port_mix_LP(w_final: pd.DataFrame, l_proxy: pd.DataFrame,
                      tons_pm: pd.DataFrame, teu_pm: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Π: quarter-constant terminal shares * terminal π_i_y
    lp = l_proxy.copy()
    lp["quarter"] = lp["month"].apply(_quarter_from_month)

    teui = (lp.groupby(["port","terminal","year","quarter"], dropna=False)["teu_i_m"]
              .sum(min_count=1).reset_index().rename(columns={"teu_i_m":"teu_i_q_sum"}))
    teutot = (teui.groupby(["port","year","quarter"], dropna=False)["teu_i_q_sum"]
                 .sum(min_count=1).reset_index().rename(columns={"teu_i_q_sum":"teu_port_q"}))
    shares = teui.merge(teutot, on=["port","year","quarter"], how="left")
    shares["share_i_q"] = np.where(shares["teu_port_q"]>0, shares["teu_i_q_sum"]/shares["teu_port_q"], np.nan)
    pi_i_y = (lp.groupby(["port","terminal","year"], dropna=False)["pi_teu_per_hour_i_y"]
                .first().reset_index())
    shares = shares.merge(pi_i_y, on=["port","terminal","year"], how="left")
    pi_port_q = (shares.assign(pi_weighted=lambda d: d["share_i_q"]*d["pi_teu_per_hour_i_y"])
                      .groupby(["port","year","quarter"], dropna=False)["pi_weighted"]
                      .sum(min_count=1).reset_index().rename(columns={"pi_weighted":"Pi_p_q"}))

    months = w_final[["port","year","month","month_index"]].drop_duplicates()
    months["quarter"] = months["month"].apply(_quarter_from_month)
    pi_pm = months.merge(pi_port_q, on=["port","year","quarter"], how="left").rename(columns={"Pi_p_q":"pi_p_y_mixbase"})

    # LP mix at port-month
    lp_port = w_final.merge(pi_pm[["port","year","month","pi_p_y_mixbase"]], on=["port","year","month"], how="left")
    lp_port["lp_port_month_mix"] = lp_port["w_final"] * lp_port["pi_p_y_mixbase"]

    # Diagnostics: add TEU and tons for transparency
    diag = tons_pm.merge(teu_pm, on=["port","year","month","month_index"], how="left")
    lp_port = lp_port.merge(diag[["port","year","month","month_index","tons_p_m","teu_p_m","tons_source"]],
                            on=["port","year","month","month_index"], how="left")

    # Identity LP (sparse post-reform)
    L_port_m = (l_proxy.groupby(["port","year","month"], dropna=False)["l_hours_i_m"]
                        .sum(min_count=1).reset_index().rename(columns={"l_hours_i_m":"l_port_m"}))
    lp_id = L_port_m.merge(teu_pm, on=["port","year","month"], how="left")
    lp_id["lp_port_month_id"] = np.where(lp_id["l_port_m"]>0, lp_id["teu_p_m"]/lp_id["l_port_m"], np.nan)

    # Merge L into lp_port for convenience
    lp_port = lp_port.merge(L_port_m, on=["port","year","month"], how="left")
    lp_port = lp_port[["port","year","month","month_index","teu_p_m","tons_p_m","w_final","w_source",
                       "pi_p_y_mixbase","lp_port_month_mix","l_port_m","tons_source"]].copy()
    lp_id = lp_id[["port","year","month","lp_port_month_id"]].copy()
    return lp_port, lp_id

def build_terminal_monthly(w_final: pd.DataFrame, l_proxy: pd.DataFrame) -> pd.DataFrame:
    df = l_proxy.merge(w_final[["port","year","month","w_final"]], on=["port","year","month"], how="left")
    df["lp_term_month_mixadjusted"] = df["pi_teu_per_hour_i_y"] * df["w_final"]
    bad = (pd.to_numeric(df["teu_i_m"], errors="coerce")<=0) | (pd.to_numeric(df["l_hours_i_m"], errors="coerce")<=0)
    df.loc[bad, "lp_term_month_mixadjusted"] = np.nan
    out = df[["port","terminal","year","month","month_index","quarter",
              "pi_teu_per_hour_i_y","w_final","teu_i_m","l_hours_i_m","lp_term_month_mixadjusted"]].copy()
    return out

def aggregate_terminals_quarter_after_cutover(term_m: pd.DataFrame, cutover: Dict[str,str]) -> pd.DataFrame:
    # Map to cutoff month_index per port; invalid -> large
    def ym_to_index(s: str) -> int:
        try:
            y, m = s.split("-")
            return int(y)*12 + int(m)
        except Exception:
            return 10**9

    cut_map = {p: ym_to_index(v) for p, v in cutover.items()}

    term = term_m.copy()
    # ensure month_index present and numeric safe
    term["month_index"] = (term["year"].astype("float")*12 + term["month"].astype("float")).round().astype("Int64")
    term["quarter"] = term["month"].apply(_quarter_from_month)
    # freq marker
    term["freq"] = np.where(term["port"].map(cut_map).astype("Int64").lt(term["month_index"]), "Q", "M")

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
        agg["month_index"] = (agg["year"].astype("float")*12 + agg["month"].astype("float")).round().astype("Int64")
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

def build_panel_mixedfreq(lp_port: pd.DataFrame, lp_id: pd.DataFrame,
                          term_m: pd.DataFrame, term_qview: pd.DataFrame) -> pd.DataFrame:
    port = lp_port.copy()
    port["level"] = "port"; port["terminal"] = pd.NA
    port["Pi"] = port["pi_p_y_mixbase"]
    port["L_hours"] = port["l_port_m"]
    port["LP_mix"] = port["lp_port_month_mix"]
    port = port.merge(lp_id, on=["port","year","month"], how="left")
    port = port.rename(columns={"lp_port_month_id":"LP_id"})
    port["quarter"] = port["month"].apply(_quarter_from_month)
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

def run_qa(lp_port: pd.DataFrame, term_m: pd.DataFrame, w_final: pd.DataFrame,
           min_port_year_coverage: float = 0.5) -> pd.DataFrame:
    rows = []
    def assert_unique(df, keys, name):
        c = df.duplicated(keys).sum()
        rows.append({"check":f"unique_keys_{name}", "result":"pass" if c==0 else "fail", "detail":f"duplicates={int(c)} keys={keys}"})
    assert_unique(lp_port, ["port","year","month"], "lp_port")
    assert_unique(term_m, ["port","terminal","year","month"], "lp_term_monthly")
    assert_unique(w_final, ["port","year","month"], "w_final")

    # Annual preservation
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

    # Coverage check
    cov = lp_port.assign(ok=lp_port["lp_port_month_mix"].notna()) \
                 .groupby(["port","year"], dropna=False)["ok"].mean().reset_index(name="coverage")
    for _, r in cov.iterrows():
        rows.append({"check":"coverage","port":r["port"],"year":int(r["year"]), "coverage":float(r["coverage"]),
                     "result":"pass" if r["coverage"]>=min_port_year_coverage else "fail"})
    return pd.DataFrame(rows)

# ------------------------------- Main ----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=".", help="Base directory of the repo; defaults to CWD.")
    ap.add_argument("--l_proxy", default=None, help="Path to L_Proxy.tsv (terminal×month).")
    ap.add_argument("--tons", default=None, help="Path to monthly_output_by_1000_tons_ports_and_terminals.tsv.")
    ap.add_argument("--teu_mq", default=None, help="Path to teu_monthly_plus_quarterly_by_port.tsv.")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: Data/LP under base_dir).")
    ap.add_argument("--columns_map", default=None, help="Path to columns_map.json (optional but recommended).")
    ap.add_argument("--cutover", default=None, help="Comma list like 'Haifa:2021-09,Ashdod:2022-07' (fallback).")
    ap.add_argument("--winsor_lower", type=float, default=0.01, help="Lower winsor quantile for r.")
    ap.add_argument("--winsor_upper", type=float, default=0.99, help="Upper winsor quantile for r.")
    ap.add_argument("--allocate_allports", action="store_true", help="Allocate All-Ports tons to ports by TEU shares if port totals and terminal sums both missing.")
    ap.add_argument("--min_port_year_coverage", type=float, default=0.5, help="Fail if LP_port_month_mix coverage below this share.")
    ap.add_argument("--validate-only", action="store_true", help="Run validations only; do not write outputs.")
    args = ap.parse_args()

    try:
        inp = load_inputs(args)
        columns_map = _read_columns_map(inp.columns_map_path)

        # Load inputs
        l_proxy = load_L_proxy(inp.l_proxy_path, columns_map)
        teu_pm, teu_pq = load_teu(inp.teu_mq_path, columns_map)
        tons_port_m, tons_term_m, tons_allports_m = load_tons(inp.tons_path, columns_map, inp.allocate_allports, teu_pm, l_proxy)

        # Crosswalk
        crosswalk_ports(l_proxy, tons_port_m, teu_pm, teu_pq)

        # Validate
        ok, report = validate_inputs(l_proxy, tons_port_m, teu_pm, teu_pq)
        print("VALIDATION REPORT:\n" + report)
        if not ok:
            sys.exit(1)
        if args.validate_only:
            print("Validation-only mode requested. Exiting without writing outputs.")
            sys.exit(0)

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

        # QA + fail-fast coverage report (non-fatal; we only print)
        qa = run_qa(lp_port, term_m, w_final, min_port_year_coverage=inp.min_port_year_coverage)

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
            "inputs": {
                "l_proxy": inp.l_proxy_path,
                "tons": inp.tons_path,
                "teu_mq": inp.teu_mq_path,
                "columns_map": inp.columns_map_path,
            },
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
