#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_lp_mixedfreq_final_v3.py
Robust LP panel builder with mixed-frequency w (monthly with quarterly fallback).

Key fixes vs prior versions:
- Never assumes month_index exists after merges; always (re)computes it.
- Enforces quarter as 'Q1'..'Q4' strings (object) across all tables before merging.
- Coerces numerics safely; provenance columns kept as object dtype.
- Handles sparse TEU monthly by broadcasting quarterly w to months present in tons.
- Friendly validation & QA outputs.
"""

import argparse, json, os, re, sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------- Utilities ---------------------------------------

class ValidationError(Exception):
    pass

def _read_tsv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t", engine="python")
    except Exception as e:
        raise ValidationError(f"Failed reading TSV: {path} :: {e}")

def _norm_port(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().replace("–","-")
    low = s.lower()
    if low.startswith("ashdod"): return "Ashdod"
    if low.startswith("haifa"):  return "Haifa"
    if low.startswith("eilat"):  return "Eilat"
    if low in {"all ports","all_ports","allports","all"}: return "All Ports"
    return s

def _q_from_month(m) -> Optional[str]:
    if pd.isna(m): return None
    try:
        q = (int(m)-1)//3 + 1
        return f"Q{q}"
    except Exception:
        return None

def _q_parse(v) -> Optional[str]:
    if pd.isna(v): return None
    s = str(v).upper().strip()
    m = re.search(r"Q([1-4])", s)
    if m: return f"Q{m.group(1)}"
    # Fallback: bare 1..4
    if s.isdigit() and 1 <= int(s) <= 4:
        return f"Q{int(s)}"
    return None

def _month_index_from_cols(df: pd.DataFrame, year_col="year", month_col="month") -> pd.Series:
    y = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    m = pd.to_numeric(df[month_col], errors="coerce").astype("Int64")
    return (y * 12 + m).astype("Int64")

def _apply_columns_map(df: pd.DataFrame, file_basename: str, columns_map: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    mp = columns_map.get(file_basename) or columns_map.get(os.path.basename(file_basename)) or {}
    if not mp: return df
    ren = {}
    for canonical, actual in mp.items():
        if actual in df.columns:
            ren[actual] = canonical
        else:
            # case-insensitive
            for c in df.columns:
                if c.lower() == str(actual).lower():
                    ren[c] = canonical
                    break
    if ren:
        df = df.rename(columns=ren)
    return df

# ----------------------- Loaders ---------------------------------------------

@dataclass
class Inputs:
    l_proxy_path: str
    tons_path: str
    teu_mq_path: str
    out_dir: str
    winsor_lower: float
    winsor_upper: float
    cutover: Dict[str, str]
    columns_map_path: Optional[str]

def _find(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p): return p
    return None

def load_inputs(args) -> Inputs:
    base = args.base_dir
    l_proxy = _find([args.l_proxy, os.path.join(base,"Data","LP","L_Proxy.tsv"), os.path.join(base,"Data","L_proxy","L_Proxy.tsv"), "L_Proxy.tsv"])
    tons    = _find([args.tons,    os.path.join(base,"Data","LP","monthly_output_by_1000_tons_ports_and_terminals.tsv"),
                     os.path.join(base,"Data","Output","monthly_output_by_1000_tons_ports_and_terminals.tsv"),
                     "monthly_output_by_1000_tons_ports_and_terminals.tsv"])
    teu_mq  = _find([args.teu_mq,  os.path.join(base,"Data","LP","teu_monthly_plus_quarterly_by_port.tsv"),
                     os.path.join(base,"Data","Output","teu_monthly_plus_quarterly_by_port.tsv"),
                     "teu_monthly_plus_quarterly_by_port.tsv"])
    if not l_proxy: raise ValidationError("Missing L_Proxy.tsv (use --l_proxy)")
    if not tons:    raise ValidationError("Missing tons file (use --tons)")
    if not teu_mq:  raise ValidationError("Missing TEU file (use --teu_mq)")
    out_dir = args.out_dir or os.path.join(base,"Data","LP")
    os.makedirs(out_dir, exist_ok=True)
    cut = {}
    if args.cutover:
        for kv in str(args.cutover).split(","):
            if ":" in kv:
                k,v = kv.split(":",1); cut[k.strip()] = v.strip()
    if not cut:
        cut = {"Haifa":"2021-09","Ashdod":"2022-07"}
    return Inputs(
        l_proxy_path=l_proxy, tons_path=tons, teu_mq_path=teu_mq, out_dir=out_dir,
        winsor_lower=float(args.winsor_lower), winsor_upper=float(args.winsor_upper),
        cutover=cut, columns_map_path=args.columns_map
    )

def load_L_proxy(path: str, columns_map: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    df = _read_tsv(path)
    df = _apply_columns_map(df, os.path.basename(path), columns_map)
    for need in ["port","terminal","year","month","l_hours_i_m","teu_i_m","pi_teu_per_hour_i_y"]:
        if need not in df.columns:
            raise ValidationError(f"L_Proxy missing column '{need}'. Map it in columns_map.json.")
    out = pd.DataFrame({
        "port": df["port"].map(_norm_port),
        "terminal": df["terminal"].astype(str).str.strip(),
        "year": pd.to_numeric(df["year"], errors="coerce").astype("Int64"),
        "month": pd.to_numeric(df["month"], errors="coerce").astype("Int64"),
        "l_hours_i_m": pd.to_numeric(df["l_hours_i_m"], errors="coerce"),
        "teu_i_m": pd.to_numeric(df["teu_i_m"], errors="coerce"),
        "pi_teu_per_hour_i_y": pd.to_numeric(df["pi_teu_per_hour_i_y"], errors="coerce"),
    })
    if "quarter" in df.columns:
        out["quarter"] = df["quarter"].apply(_q_parse).astype("object")
    else:
        out["quarter"] = out["month"].apply(_q_from_month).astype("object")
    out["month_index"] = _month_index_from_cols(out)
    return out

def load_tons_ports_and_terminals(path: str, columns_map: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    df = _read_tsv(path)
    df = _apply_columns_map(df, os.path.basename(path), columns_map)
    for need in ["port","year","month","tons"]:
        if need not in df.columns:
            raise ValidationError(f"Tons file missing '{need}'. Map it in columns_map.json.")
    tmp = pd.DataFrame({
        "port": df["port"].map(_norm_port),
        "terminal": df["terminal"].astype(str).str.strip() if "terminal" in df.columns else pd.NA,
        "year": pd.to_numeric(df["year"], errors="coerce").astype("Int64"),
        "month": pd.to_numeric(df["month"], errors="coerce").astype("Int64"),
        "tons": pd.to_numeric(df["tons"], errors="coerce"),
    })
    # scale if provided in thousands
    name_l = "tons"
    if "tons_unit" in df.columns and str(df["tons_unit"].iloc[0]).lower() in ["k","1000","thousand"]:
        tmp["tons"] = tmp["tons"] * 1000.0
    # prefer port totals; else sum terminals
    is_all = tmp["port"].astype(str).str.lower().isin(["all ports","all_ports","allports","all"])
    tmp = tmp.loc[~is_all].copy()
    is_port_total = tmp["terminal"].isna() | (tmp["terminal"].astype(str).str.strip()=="") | (tmp["terminal"].astype(str).str.lower().isin(["nan","none","na"]))
    tons_port = tmp.loc[is_port_total, ["port","year","month","tons"]].copy()
    tons_port["tons_source"] = "port_total"
    tons_term = tmp.loc[~is_port_total, ["port","terminal","year","month","tons"]].copy()
    # sum terminals to get fallback port totals
    if not tons_term.empty:
        sum_term = tons_term.groupby(["port","year","month"], dropna=False)["tons"].sum(min_count=1).reset_index()
        merged = tons_port.merge(sum_term, on=["port","year","month"], how="outer", suffixes=("_p","_sum"))
        merged["tons_p_m"] = merged["tons_p"].combine_first(merged["tons_sum"])
        merged["tons_source"] = merged["tons_source"].fillna(np.where(merged["tons_sum"].notna(),"sum_terminals",np.nan)).astype("object")
        tons_port_m = merged[["port","year","month","tons_p_m","tons_source"]].copy()
    else:
        tons_port_m = tons_port.rename(columns={"tons":"tons_p_m"})[["port","year","month","tons_p_m","tons_source"]].copy()
    tons_port_m["month_index"] = _month_index_from_cols(tons_port_m)
    return tons_port_m, tons_term

def load_teu_monthly_quarterly_by_port(path: str, columns_map: Dict[str, Dict[str, str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = _read_tsv(path)
    df = _apply_columns_map(df, os.path.basename(path), columns_map)
    if "port" not in df.columns or "year" not in df.columns:
        raise ValidationError("TEU file needs 'port' and 'year' (map via columns_map.json).")
    vcol = "teu" if "teu" in df.columns else None
    if not vcol:
        for c in df.columns:
            if "teu" in c.lower() or c.lower() in ["value","count","qty"]:
                vcol = c; break
    if not vcol:
        raise ValidationError("TEU file: could not find the TEU value column.")
    out = df.copy()
    out["port"] = out["port"].map(_norm_port)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    teu_m = pd.DataFrame(columns=["port","year","month","teu_p_m"])
    if "month" in out.columns:
        mp = out[out["month"].notna()].copy()
        if not mp.empty:
            mp["month"] = pd.to_numeric(mp["month"], errors="coerce").astype("Int64")
            teu_m = mp[["port","year","month",vcol]].rename(columns={vcol:"teu_p_m"})
            # month_index not strictly needed; recompute when needed
    teu_q = pd.DataFrame(columns=["port","year","quarter","teu_p_q"])
    qcol = "quarter" if "quarter" in out.columns else None
    if qcol:
        qp = out[out[qcol].notna()].copy()
        if not qp.empty:
            qp["quarter"] = qp[qcol].apply(_q_parse).astype("object")
            teu_q = qp[["port","year","quarter",vcol]].rename(columns={vcol:"teu_p_q"})
    else:
        # look for 'period' style
        per = None
        for c in out.columns:
            if c.lower() in ["period","year_quarter","yr_qtr","yyyyq","yyyq","yyyyqq"]:
                per = c; break
        if per:
            qp = out[out[per].notna()].copy()
            qp["quarter"] = qp[per].apply(_q_parse).astype("object")
            yr_guess = pd.to_numeric(qp[per].astype(str).str.extract(r"(\d{4})")[0], errors="coerce").astype("Int64")
            qp["year"] = qp["year"].fillna(yr_guess).astype("Int64")
            teu_q = qp[["port","year","quarter",vcol]].rename(columns={vcol:"teu_p_q"})
    return teu_m, teu_q

# -------------------------- Validation ---------------------------------------

def validate_inputs(l_proxy: pd.DataFrame, tons_port_m: pd.DataFrame, teu_pm: pd.DataFrame, teu_pq: pd.DataFrame) -> Tuple[bool,str]:
    msgs = []
    for col in ["port","terminal","year","month","l_hours_i_m","teu_i_m","pi_teu_per_hour_i_y"]:
        if col not in l_proxy.columns: msgs.append(f"[L_Proxy] Missing {col}")
    if l_proxy.duplicated(["port","terminal","year","month"]).any():
        msgs.append("[L_Proxy] Duplicate (port,terminal,year,month)")
    for col in ["port","year","month","tons_p_m"]:
        if col not in tons_port_m.columns: msgs.append(f"[Tons] Missing {col}")
    if tons_port_m.duplicated(["port","year","month"]).any():
        msgs.append("[Tons] Duplicate (port,year,month)")
    ok = len(msgs)==0
    return ok, ("All validations passed." if ok else "\n".join(msgs))

# -------------------------- Core: compute w ----------------------------------

def winsorize_group(df: pd.DataFrame, value_col: str, by: List[str], lower=0.01, upper=0.99) -> pd.Series:
    vals = pd.to_numeric(df[value_col], errors="coerce")
    g = df.assign(_val=vals).groupby(by, dropna=False, sort=False)["_val"]
    qs = g.quantile([lower, upper]).unstack(level=-1)
    if qs is None or qs.empty:
        return vals
    lo = qs[lower]; hi = qs[upper]
    # align to rows
    key = pd.MultiIndex.from_frame(df[by])
    lo = lo.reindex(key).reset_index(drop=True)
    hi = hi.reindex(key).reset_index(drop=True)
    v = vals.to_numpy(dtype="float64")
    vlo = lo.to_numpy(dtype="float64"); vhi = hi.to_numpy(dtype="float64")
    v = np.where(~np.isnan(v) & ~np.isnan(vlo), np.maximum(v, vlo), v)
    v = np.where(~np.isnan(v) & ~np.isnan(vhi), np.minimum(v, vhi), v)
    return pd.Series(v, index=df.index)

def compute_w(tons_pm: pd.DataFrame, teu_pm: pd.DataFrame, teu_pq: pd.DataFrame,
              lower=0.01, upper=0.99) -> pd.DataFrame:
    # monthly path
    w_m = tons_pm.merge(teu_pm, on=["port","year","month"], how="left")
    w_m["tons_per_teu"] = np.where(pd.to_numeric(w_m.get("teu_p_m"), errors="coerce")>0,
                                   pd.to_numeric(w_m["tons_p_m"], errors="coerce")/pd.to_numeric(w_m["teu_p_m"], errors="coerce"),
                                   np.nan)
    w_m["r_win"] = winsorize_group(w_m, "tons_per_teu", by=["port","year"], lower=lower, upper=upper)
    mean_by = w_m.groupby(["port","year"], dropna=False)["r_win"].transform("mean")
    w_m["w_p_m"] = np.where((mean_by==0) | (mean_by.isna()), 1.0, w_m["r_win"]/mean_by)
    w_m["w_src_monthly"] = pd.Series(np.where(w_m["tons_per_teu"].notna(),"monthly",None), index=w_m.index, dtype="object")
    # ensure quarter + month_index exist
    w_m["month_index"] = _month_index_from_cols(w_m)
    w_m["quarter"] = w_m["month"].apply(_q_from_month).astype("object")

    # quarterly fallback
    if teu_pq.empty:
        # No quarterly TEU: leave fallback empty
        w_qm = w_m[["port","year","month"]].copy()
        w_qm["w_from_q"] = np.nan
        w_qm["w_src_quarterly"] = pd.Series([None]*len(w_qm), dtype="object")
    else:
        # Aggregate tons to port-year-quarter
        temp = tons_pm.copy()
        temp["quarter"] = temp["month"].apply(_q_from_month).astype("object")
        agg = temp.groupby(["port","year","quarter"], dropna=False)["tons_p_m"].sum(min_count=1).reset_index()
        tq = teu_pq.copy()
        tq["quarter"] = tq["quarter"].apply(_q_parse).astype("object")
        rq = agg.merge(tq, on=["port","year","quarter"], how="left")
        rq["r_q"] = np.where(pd.to_numeric(rq.get("teu_p_q"), errors="coerce")>0,
                             pd.to_numeric(rq["tons_p_m"], errors="coerce")/pd.to_numeric(rq["teu_p_q"], errors="coerce"),
                             np.nan)
        rq["r_q_win"] = winsorize_group(rq, "r_q", by=["port","year"], lower=lower, upper=upper)
        mean_q = rq.groupby(["port","year"], dropna=False)["r_q_win"].transform("mean")
        rq["w_p_q"] = np.where((mean_q==0) | (mean_q.isna()), 1.0, rq["r_q_win"]/mean_q)

        # Broadcast quarterly w to months present in tons
        months = tons_pm[["port","year","month"]].drop_duplicates().copy()
        months["quarter"] = months["month"].apply(_q_from_month).astype("object")
        w_qm = months.merge(rq[["port","year","quarter","w_p_q"]], on=["port","year","quarter"], how="left").rename(columns={"w_p_q":"w_from_q"})
        w_qm["w_src_quarterly"] = pd.Series(np.where(w_qm["w_from_q"].notna(),"quarterly",None), dtype="object")

    # Final w: merge on port/year/month (recompute index post-merge)
    wf = w_m[["port","year","month","w_p_m","w_src_monthly"]].merge(
        w_qm[["port","year","month","w_from_q","w_src_quarterly"]],
        on=["port","year","month"], how="outer"
    )
    wf["w_final"] = wf["w_p_m"].combine_first(wf["w_from_q"])
    wf["w_source"] = wf["w_src_monthly"].combine_first(wf["w_src_quarterly"]).astype("object")
    # Recompute month_index deterministically
    wf["month_index"] = _month_index_from_cols(wf)
    return wf[["port","year","month","month_index","w_final","w_source"]].copy()

# ------------------- LP construction -----------------------------------------

def build_port_mix_LP(w_final: pd.DataFrame, l_proxy: pd.DataFrame,
                      tons_pm: pd.DataFrame, teu_pm: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lp = l_proxy.copy()
    # terminal shares by port-quarter
    lp["quarter"] = lp["month"].apply(_q_from_month).astype("object")
    teui = lp.groupby(["port","terminal","year","quarter"], dropna=False)["teu_i_m"].sum(min_count=1).reset_index().rename(columns={"teu_i_m":"teu_i_q_sum"})
    teutot = teui.groupby(["port","year","quarter"], dropna=False)["teu_i_q_sum"].sum(min_count=1).reset_index().rename(columns={"teu_i_q_sum":"teu_port_q"})
    shares = teui.merge(teutot, on=["port","year","quarter"], how="left")
    shares["share_i_q"] = np.where(pd.to_numeric(shares["teu_port_q"], errors="coerce")>0,
                                   pd.to_numeric(shares["teu_i_q_sum"], errors="coerce")/pd.to_numeric(shares["teu_port_q"], errors="coerce"),
                                   np.nan)
    pi_i_y = lp.groupby(["port","terminal","year"], dropna=False)["pi_teu_per_hour_i_y"].first().reset_index()
    shares = shares.merge(pi_i_y, on=["port","terminal","year"], how="left")
    pi_port_q = shares.assign(pi_w=lambda d: d["share_i_q"]*d["pi_teu_per_hour_i_y"]).groupby(["port","year","quarter"], dropna=False)["pi_w"].sum(min_count=1).reset_index().rename(columns={"pi_w":"Pi_p_q"})
    # broadcast to months from w_final keys
    months = w_final[["port","year","month"]].drop_duplicates().copy()
    months["quarter"] = months["month"].apply(_q_from_month).astype("object")
    pi_pm = months.merge(pi_port_q, on=["port","year","quarter"], how="left").rename(columns={"Pi_p_q":"pi_p_y_mixbase"})

    # LP mix at port-month
    lp_port = w_final.merge(pi_pm, on=["port","year","month"], how="left")
    lp_port["lp_port_month_mix"] = pd.to_numeric(lp_port["w_final"], errors="coerce") * pd.to_numeric(lp_port["pi_p_y_mixbase"], errors="coerce")

    # Diagnostics: add tons & monthly TEU where available
    diag = tons_pm.merge(teu_pm, on=["port","year","month"], how="left")
    diag["month_index"] = _month_index_from_cols(diag)
    lp_port = lp_port.merge(diag[["port","year","month","month_index","tons_p_m","teu_p_m"]], on=["port","year","month","month_index"], how="left")
    # Add L
    L_port_m = l_proxy.groupby(["port","year","month"], dropna=False)["l_hours_i_m"].sum(min_count=1).reset_index().rename(columns={"l_hours_i_m":"l_port_m"})
    lp_port = lp_port.merge(L_port_m, on=["port","year","month"], how="left")

    # Identity LP
    lp_id = L_port_m.merge(teu_pm, on=["port","year","month"], how="left")
    lp_id["lp_port_month_id"] = np.where(pd.to_numeric(lp_id["l_port_m"], errors="coerce")>0,
                                         pd.to_numeric(lp_id["teu_p_m"], errors="coerce")/pd.to_numeric(lp_id["l_port_m"], errors="coerce"),
                                         np.nan)

    # Final tidy
    lp_port["month_index"] = _month_index_from_cols(lp_port)
    lp_port = lp_port[["port","year","month","month_index","teu_p_m","tons_p_m","w_final","w_source","pi_p_y_mixbase","lp_port_month_mix","l_port_m"]].copy()
    lp_id = lp_id[["port","year","month","lp_port_month_id"]].copy()
    return lp_port, lp_id

def build_terminal_monthly(w_final: pd.DataFrame, l_proxy: pd.DataFrame) -> pd.DataFrame:
    df = l_proxy.merge(w_final[["port","year","month","w_final"]], on=["port","year","month"], how="left")
    df["lp_term_month_mixadjusted"] = pd.to_numeric(df["pi_teu_per_hour_i_y"], errors="coerce") * pd.to_numeric(df["w_final"], errors="coerce")
    bad = (pd.to_numeric(df["teu_i_m"], errors="coerce")<=0) | (pd.to_numeric(df["l_hours_i_m"], errors="coerce")<=0)
    df.loc[bad, "lp_term_month_mixadjusted"] = np.nan
    df["month_index"] = _month_index_from_cols(df)
    df["quarter"] = df["month"].apply(_q_from_month).astype("object")
    keep = ["port","terminal","year","month","month_index","quarter","pi_teu_per_hour_i_y","w_final","teu_i_m","l_hours_i_m","lp_term_month_mixadjusted"]
    return df[keep].copy()

def aggregate_terminals_quarter_after_cutover(term_m: pd.DataFrame, cutover: Dict[str,str]) -> pd.DataFrame:
    # determine per-port cutover index (YYYY-MM -> index)
    def idx_from_ym(s: str) -> int:
        y,m = s.split("-"); return int(y)*12 + int(m)
    cut_map = {p: idx_from_ym(v) for p,v in cutover.items()}
    t = term_m.copy()
    t["month_index"] = _month_index_from_cols(t)
    t["freq"] = np.where(t["port"].map(cut_map).le(t["month_index"]), "Q", "M")
    q_to_month = {"Q1":3,"Q2":6,"Q3":9,"Q4":12}

    term_M = t[t["freq"]=="M"].copy()
    term_M["freq"] = "M"

    term_Q = t[t["freq"]=="Q"].copy()
    if term_Q.empty:
        out_Q = term_Q.copy()
    else:
        agg = term_Q.groupby(["port","terminal","year","quarter"], dropna=False).agg(
            pi_teu_per_hour_i_y=("pi_teu_per_hour_i_y","first"),
            w_final=("w_final","mean"),
            teu_i_m=("teu_i_m","sum"),
            l_hours_i_m=("l_hours_i_m","sum"),
            lp_term_month_mixadjusted=("lp_term_month_mixadjusted","mean"),
        ).reset_index()
        agg["month"] = agg["quarter"].map(q_to_month).astype("Int64")
        agg["month_index"] = _month_index_from_cols(agg)
        agg["freq"] = "Q"
        out_Q = agg

    cols = ["port","terminal","year","quarter","month","month_index","freq","pi_teu_per_hour_i_y","w_final","teu_i_m","l_hours_i_m","lp_term_month_mixadjusted"]
    out = pd.concat([term_M[cols], out_Q[cols]], ignore_index=True).sort_values(["port","terminal","year","month"]).reset_index(drop=True)
    return out

def build_panel(lp_port: pd.DataFrame, lp_id: pd.DataFrame, term_m: pd.DataFrame, term_qview: pd.DataFrame) -> pd.DataFrame:
    port = lp_port.copy()
    port["level"] = "port"; port["terminal"] = pd.NA
    port["Pi"] = port["pi_p_y_mixbase"]; port["L_hours"] = port["l_port_m"]
    port["LP_mix"] = port["lp_port_month_mix"]; port = port.merge(lp_id, on=["port","year","month"], how="left")
    port = port.rename(columns={"lp_port_month_id":"LP_id"})
    port["quarter"] = port["month"].apply(_q_from_month).astype("object")
    port["TEU"] = port["teu_p_m"]; port["tons"] = port["tons_p_m"]
    port["w"] = port["w_final"]; port["w_source"] = port["w_source"].astype("object")
    port["freq"] = "M"
    port = port[["level","port","terminal","year","month","month_index","quarter","freq","TEU","tons","w","w_source","Pi","L_hours","LP_mix","LP_id"]]

    term = term_qview.copy()
    term["level"] = "terminal"
    term = term.rename(columns={"pi_teu_per_hour_i_y":"Pi", "l_hours_i_m":"L_hours", "lp_term_month_mixadjusted":"LP_mix", "teu_i_m":"TEU", "w_final":"w"})
    term["LP_id"] = pd.NA; term["tons"] = pd.NA; term["w_source"] = pd.NA
    term = term[["level","port","terminal","year","month","month_index","quarter","freq","TEU","tons","w","w_source","Pi","L_hours","LP_mix","LP_id"]]

    panel = pd.concat([port, term], ignore_index=True).sort_values(["level","port","terminal","year","month"]).reset_index(drop=True)
    return panel

def run_qa(lp_port: pd.DataFrame, term_m: pd.DataFrame, w_final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(check, **kw): rows.append({"check":check, **kw})
    for name, df, keys in [
        ("lp_port", lp_port, ["port","year","month"]),
        ("lp_term_monthly", term_m, ["port","terminal","year","month"]),
        ("w_final", w_final, ["port","year","month"]),
    ]:
        dups = int(df.duplicated(keys).sum())
        add(f"unique_keys_{name}", result="pass" if dups==0 else "fail", detail=f"duplicates={dups}")
    # w_source shares
    src = w_final.assign(w_source=w_final["w_source"].astype("object")).groupby(["port","year","w_source"], dropna=False).size().reset_index(name="n")
    total = w_final.groupby(["port","year"], dropna=False).size().reset_index(name="N")
    src = src.merge(total, on=["port","year"], how="left")
    src["share"] = src["n"]/src["N"]
    for _, r in src.iterrows():
        add("w_source_share", port=r["port"], year=int(r["year"]), w_source=None if pd.isna(r["w_source"]) else str(r["w_source"]), n=int(r["n"]), N=int(r["N"]), share=float(r["share"]))
    return pd.DataFrame(rows)

# -------------------------------- Main ---------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=".")
    ap.add_argument("--l_proxy", default=None)
    ap.add_argument("--tons", default=None)
    ap.add_argument("--teu_mq", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--columns_map", default=None, help="Path to columns_map.json (recommended)")
    ap.add_argument("--cutover", default=None, help="e.g., 'Haifa:2021-09,Ashdod:2022-07'")
    ap.add_argument("--winsor_lower", type=float, default=0.01)
    ap.add_argument("--winsor_upper", type=float, default=0.99)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    try:
        inp = load_inputs(args)
        columns_map = {}
        if inp.columns_map_path and os.path.exists(inp.columns_map_path):
            with open(inp.columns_map_path, "r", encoding="utf-8") as f:
                columns_map = json.load(f)

        l_proxy = load_L_proxy(inp.l_proxy_path, columns_map)
        tons_port_m, tons_term_m = load_tons_ports_and_terminals(inp.tons_path, columns_map)
        teu_pm, teu_pq = load_teu_monthly_quarterly_by_port(inp.teu_mq_path, columns_map)

        ok, rep = validate_inputs(l_proxy, tons_port_m, teu_pm, teu_pq)
        print("VALIDATION REPORT:\n" + rep)
        if not ok:
            sys.exit(1)
        if args.validate_only:
            print("Validation-only; exiting.")
            sys.exit(0)

        w_final = compute_w(tons_port_m, teu_pm, teu_pq, lower=inp.winsor_lower, upper=inp.winsor_upper)
        lp_port, lp_id = build_port_mix_LP(w_final, l_proxy, tons_port_m, teu_pm)
        term_m = build_terminal_monthly(w_final, l_proxy)
        term_q = aggregate_terminals_quarter_after_cutover(term_m, inp.cutover)
        panel = build_panel(lp_port, lp_id, term_m, term_q)
        qa = run_qa(lp_port, term_m, w_final)

        def write(df, name):
            path = os.path.join(inp.out_dir, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, sep="\t", index=False)
            return path

        write(lp_port, "LP_port_month_mixadjusted.tsv")
        write(lp_id, "LP_port_month_identity.tsv")
        write(term_m, "LP_terminal_month_mixadjusted.tsv")
        write(term_q, "LP_terminal_quarter_mixadjusted.tsv")
        write(panel, "LP_panel_mixedfreq.tsv")
        write(qa, "qa_lp_report.tsv")

        meta = {
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "inputs": {"l_proxy": inp.l_proxy_path, "tons": inp.tons_path, "teu_mq": inp.teu_mq_path, "columns_map": inp.columns_map_path},
            "cutover": inp.cutover,
            "winsor": {"lower": inp.winsor_lower, "upper": inp.winsor_upper},
            "rows": { "LP_port_month_mixadjusted": len(lp_port), "LP_port_month_identity": len(lp_id),
                      "LP_terminal_month_mixadjusted": len(term_m), "LP_terminal_quarter_mixadjusted": len(term_q),
                      "LP_panel_mixedfreq": len(panel), "qa_lp_report": len(qa) },
            "out_dir": inp.out_dir,
        }
        with open(os.path.join(inp.out_dir, "_meta_lp_mixadjusted.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print("Build completed. Outputs in:", inp.out_dir)

    except ValidationError as ve:
        print(f"[VALIDATION ERROR] {ve}"); sys.exit(1)
    except Exception as e:
        print(f"[UNEXPECTED ERROR] {e}"); sys.exit(1)

if __name__ == "__main__":
    main()
