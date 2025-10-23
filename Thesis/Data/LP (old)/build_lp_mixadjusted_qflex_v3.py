#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_lp_mixadjusted_qflex_v3.py
--------------------------------
Fixes for terminal LP completeness post-reform:
- Robust quarterly fallback for w continues
- Stronger canonicalization of port/terminal names before all merges
- Terminal LP mask now only removes **pre-opening months** per terminal
  (defined as months strictly before the first month where TEU_i_m>0 OR L_hours_i_m>0);
  post-opening months compute LP even if monthly TEU_i_m is missing.
- Ensure w_final always merges to terminal rows; includes a safety backfill
  from port-month w_final if needed.
- Extra QA table reports 2024 LP_mix non-null counts for each terminal.
"""

import argparse, json, os, hashlib, re
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
DATA_DIR = HERE.parents[1]
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
OUT_QA_TERMS_DEFAULT = LP_DIR / "qa_lp_terminals_coverage.tsv"

CANON_PORTS = ["Ashdod", "Haifa", "Eilat"]
ALLOCATION_PORTS_DEFAULT = ["Ashdod", "Haifa", "Eilat"]

# ---------------- utils ----------------

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
    if s.startswith("port of haifa"): return "Haifa"
    if s.startswith("port of ashdod"): return "Ashdod"
    if s.startswith("port of eilat"): return "Eilat"
    return str(x).strip() if not pd.isna(x) else x

def canon_terminal_name(x: str) -> str:
    s = _norm_key(x)
    if not s: return np.nan
    if ("ashdod" in s and "hct" in s) or ("southport" in s):
        return "Ashdod-HCT"
    if "ashdod" in s and "legacy" in s:
        return "Ashdod-Legacy"
    if ("haifa" in s) and ("sipg" in s or "bayport" in s):
        return "Haifa-Bayport"
    if "haifa" in s and "legacy" in s:
        return "Haifa-Legacy"
    if s == _norm_key("Ashdod-HCT"): return "Ashdod-HCT"
    if s == _norm_key("Haifa-Bayport"): return "Haifa-Bayport"
    if s == _norm_key("Ashdod-Legacy"): return "Ashdod-Legacy"
    if s == _norm_key("Haifa-Legacy"): return "Haifa-Legacy"
    if s == _norm_key("Eilat"): return "Eilat"
    return np.nan

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
                dt = pd.to_datetime(s, errors="raise")
            else:
                dt = pd.to_datetime(s, format=fmt, errors="raise")
            break
        except Exception:
            continue
    if dt is None or pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        return (np.nan, np.nan)
    return int(dt.year), int(dt.month)

def parse_quarter_to_yq(s: str) -> Tuple[int, str]:
    if pd.isna(s):
        return (np.nan, np.nan)
    raw = str(s).strip().upper()
    compact = re.sub(r"[\s_/\-]+", "", raw)  # 'Q1-2020' -> 'Q12020'

    m = re.match(r"^(?P<y>\d{4})Q(?P<q>[1-4])$", compact)
    if m:
        return int(m.group("y")), f"Q{m.group('q')}"
    m = re.match(r"^Q(?P<q>[1-4])(?P<y>\d{4})$", compact)
    if m:
        return int(m.group("y")), f"Q{m.group('q')}"
    m = re.search(r"(?P<y>\d{4}).*Q(?P<q>[1-4])", raw)
    if m:
        return int(m.group("y")), f"Q{m.group('q')}"
    m = re.search(r"Q(?P<q>[1-4]).*(?P<y>\d{4})", raw)
    if m:
        return int(m.group("y")), f"Q{m.group('q')}"

    try:
        p = pd.Period(raw.replace("-", " ").replace("/", " "), freq="Q")
        return int(p.year), f"Q{int(p.quarter)}"
    except Exception:
        return (np.nan, np.nan)

def month_to_quarter(month: int) -> str:
    return f"Q{((int(month)-1)//3)+1}"

def write_tsv(df: pd.DataFrame, path: str, force: bool=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if (not force) and os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    df.to_csv(path, index=False, sep="\t")

# ---------------- loaders ----------------

def load_teu_split(path: str):
    df = pd.read_csv(path, sep="\t|,", engine="python")
    df = normalize_columns(df)
    if "port" not in df.columns:
        raise ValueError("TEU file missing 'Port' column.")
    port = df["port"].map(canon_port_name)
    freq = df["freq"].astype(str).str.lower() if "freq" in df.columns else pd.Series(np.nan, index=df.index)

    # Monthly
    mon_mask = freq.str.contains("month", na=False) | freq.eq("m")
    dfm = df.loc[mon_mask].copy()
    if not dfm.empty:
        if "year" not in dfm.columns:
            if "period" in dfm.columns:
                y, m = zip(*dfm["period"].map(parse_month_year_to_ym))
                dfm["year"], dfm["month"] = list(y), list(m)
            elif "monthindex" in dfm.columns:
                mi = dfm["monthindex"].astype(int); dfm["year"] = mi//100; dfm["month"] = mi%100
            else:
                raise ValueError("Monthly TEU lacks Year/Month or parsable Period/MonthIndex.")
        elif "month" not in dfm.columns and "monthindex" in dfm.columns:
            mi = dfm["monthindex"].astype(int); dfm["year"] = mi//100; dfm["month"] = mi%100
        if "teu" in dfm.columns:
            dfm["teu_p_m"] = pd.to_numeric(dfm["teu"], errors="coerce")
        elif "teu_thousands" in dfm.columns:
            dfm["teu_p_m"] = pd.to_numeric(dfm["teu_thousands"], errors="coerce")*1000.0
        else:
            raise ValueError("Monthly TEU must have 'TEU' or 'TEU_thousands'.")
        dfm["port"] = port.loc[dfm.index]
        dfm = dfm.dropna(subset=["port","year","month"])
        dfm["year"] = dfm["year"].astype(int); dfm["month"] = dfm["month"].astype(int)
        dfm = dfm[dfm["port"].isin(CANON_PORTS+["All Ports"])]
        dfm = dfm.groupby(["port","year","month"], as_index=False)["teu_p_m"].sum()
        dfm["month_index"] = dfm["year"]*100 + dfm["month"]
    else:
        dfm = pd.DataFrame(columns=["port","year","month","teu_p_m","month_index"])

    # Quarterly
    qtr_mask = freq.str.contains("quarter", na=False) | freq.eq("q")
    dfq = df.loc[qtr_mask].copy()
    if not dfq.empty:
        if "period" in dfq.columns:
            yq = dfq["period"].map(parse_quarter_to_yq).tolist()
            ys = [yy for yy, _ in yq]
            qs = [qq for _, qq in yq]
            dfq["year"], dfq["quarter"] = ys, qs
        elif {"year","quarter"}.issubset(dfq.columns):
            dfq["quarter"] = dfq["quarter"].astype(str).str.upper().str.replace(" ", "")
        else:
            raise ValueError("Quarterly TEU lacks 'Period' or (Year, Quarter).")
        if "teu" in dfq.columns:
            dfq["teu_p_q"] = pd.to_numeric(dfq["teu"], errors="coerce")
        elif "teu_thousands" in dfq.columns:
            dfq["teu_p_q"] = pd.to_numeric(dfq["teu_thousands"], errors="coerce")*1000.0
        else:
            raise ValueError("Quarterly TEU must have 'TEU' or 'TEU_thousands'.")
        dfq["port"] = port.loc[dfq.index]
        dfq = dfq.dropna(subset=["port","year","quarter"])
        dfq["year"] = dfq["year"].astype(int)
        dfq = dfq[dfq["port"].isin(CANON_PORTS+["All Ports"])]
        dfq = dfq.groupby(["port","year","quarter"], as_index=False)["teu_p_q"].sum()
    else:
        dfq = pd.DataFrame(columns=["port","year","quarter","teu_p_q"])

    return dfm, dfq

def load_tons_mixed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t|,", engine="python")
    df = normalize_columns(df)
    required = ["portorterminal", "month_year", "tons_k"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Tons file missing columns: {missing}")
    y, m = zip(*df["month_year"].map(parse_month_year_to_ym))
    df["year"], df["month"] = list(y), list(m)
    df["portorterminal_raw"] = df["portorterminal"].astype(str)
    df["port_label"] = df["portorterminal_raw"].map(canon_port_name)
    df["terminal_label"] = df["portorterminal_raw"].map(canon_terminal_name)
    df["tons"] = pd.to_numeric(df["tons_k"], errors="coerce") * 1000.0
    return df

def load_l_proxy(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t|,", engine="python")
    df = normalize_columns(df)
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "teu_i_m": col_map[c] = "teu_i_m"
        elif cl in ("l_hours_i_m","l_i_m","l_hours"): col_map[c] = "l_hours_i_m"
        elif cl in ("pi_teu_per_hour_i_y","pi_i_y","pi"): col_map[c] = "pi_teu_per_hour_i_y"
        elif cl == "quarter": col_map[c] = "quarter"
        elif cl == "year": col_map[c] = "year"
        elif cl == "month": col_map[c] = "month"
        elif cl == "port": col_map[c] = "port"
        elif cl == "terminal": col_map[c] = "terminal"
    df = df.rename(columns=col_map)
    required = ["port","terminal","year","month","quarter","teu_i_m","l_hours_i_m","pi_teu_per_hour_i_y"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"L_Proxy missing required column: {r}")
    df["port"] = df["port"].map(canon_port_name)
    df["terminal"] = df["terminal"].map(lambda x: x if pd.isna(x) else str(x).strip())
    df["terminal"] = df["terminal"].map(canon_terminal_name).fillna(df["terminal"])
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    if df["quarter"].dtype != object:
        df["quarter"] = df["quarter"].astype(str)
    df["quarter"] = df["quarter"].str.upper().str.replace(" ", "")
    df["month_index"] = df["year"]*100 + df["month"]
    return df

# ---------------- build pieces ----------------

def build_tons_pm(df_mixed: pd.DataFrame, df_teu_pm: pd.DataFrame, allocation_ports: List[str]) -> pd.DataFrame:
    df_all = (
        df_mixed.loc[df_mixed["port_label"] == "All Ports", ["year","month","tons"]]
        .groupby(["year","month"], as_index=False)["tons"].sum()
    )
    df_port_rows = (
        df_mixed.loc[df_mixed["port_label"].isin(CANON_PORTS), ["port_label","year","month","tons"]]
        .rename(columns={"port_label":"port"})
        .groupby(["port","year","month"], as_index=False)["tons"].sum()
    )
    df_term_rows = df_mixed.loc[df_mixed["terminal_label"].notna()].copy()
    df_term_rows["port"] = df_term_rows["terminal_label"].map(terminal_parent_port)
    df_term_rows = (
        df_term_rows.loc[df_term_rows["port"].isin(CANON_PORTS), ["port","year","month","tons"]]
        .groupby(["port","year","month"], as_index=False)["tons"].sum()
    )

    base = df_teu_pm.loc[df_teu_pm["port"].isin(CANON_PORTS), ["port","year","month","teu_p_m"]].copy() if not df_teu_pm.empty else pd.DataFrame(columns=["port","year","month","teu_p_m"])
    merged = (
        base.merge(df_port_rows, on=["port","year","month"], how="right").rename(columns={"tons":"tons_porttotal"})
             .merge(df_term_rows, on=["port","year","month"], how="left").rename(columns={"tons":"tons_terminalsum"})
             .merge(df_all, on=["year","month"], how="left")
    )

    if not base.empty:
        alloc = (
            base.loc[base["port"].isin(ALLOCATION_PORTS_DEFAULT)]
            .groupby(["year","month"], as_index=False)["teu_p_m"].sum()
            .rename(columns={"teu_p_m":"teu_alloc_sum"})
        )
        merged = merged.merge(alloc, on=["year","month"], how="left")
    else:
        merged["teu_alloc_sum"] = np.nan
        merged["teu_p_m"] = merged.get("teu_p_m", np.nan)

    def decide_tons(row):
        if not pd.isna(row.get("tons_porttotal")):
            return row["tons_porttotal"], "port_total"
        if not pd.isna(row.get("tons_terminalsum")):
            return row["tons_terminalsum"], "sum_terminals"
        if (not pd.isna(row.get("tons"))) and (row["port"] in ALLOCATION_PORTS_DEFAULT) and (row.get("teu_alloc_sum",0) not in (0,np.nan)):
            share = (row["teu_p_m"]/row["teu_alloc_sum"]) if row["teu_alloc_sum"] else np.nan
            return row["tons"]*share, "allocated_allports"
        return np.nan, "no_source"

    tons_list, src_list = [], []
    for _, r in merged.iterrows():
        t, s = decide_tons(r)
        tons_list.append(t); src_list.append(s)

    merged["tons_p_m"] = tons_list
    merged["tons_source"] = src_list
    merged["month_index"] = merged["year"]*100 + merged["month"]
    return merged[["port","year","month","month_index","teu_p_m","tons_p_m","tons_source"]]

def winsorize_group(df: pd.DataFrame, value_col: str, by: List[str], lower=0.01, upper=0.99) -> pd.Series:
    def _clip(s: pd.Series):
        x = pd.to_numeric(s, errors="coerce").astype(float)
        finite = np.isfinite(x)
        if finite.sum()==0:
            return x*np.nan
        lo = np.nanquantile(x[finite], lower); hi = np.nanquantile(x[finite], upper)
        y = x.copy(); y[finite] = np.clip(x[finite], lo, hi)
        return y
    return df.groupby(by, group_keys=False)[value_col].apply(_clip)

def compute_w_monthly(df_tons_pm: pd.DataFrame) -> pd.DataFrame:
    df = df_tons_pm.copy()
    df["tons_per_teu"] = np.where(df["teu_p_m"]>0, df["tons_p_m"]/df["teu_p_m"], np.nan)
    df["r_winsor"] = winsorize_group(df, "tons_per_teu", by=["port","year"], lower=0.01, upper=0.99)
    mean_by_py = df.groupby(["port","year"])["r_winsor"].transform("mean")
    df["w_p_m"] = np.where((mean_by_py==0) | (mean_by_py.isna()), 1.0, df["r_winsor"]/mean_by_py)
    df["w_p_m"] = df["w_p_m"].fillna(1.0)
    return df[["port","year","month","month_index","w_p_m","tons_per_teu"]]

def compute_w_quarter_fallback(df_tons_pm: pd.DataFrame, df_teu_pq: pd.DataFrame) -> pd.DataFrame:
    if df_teu_pq.empty:
        return pd.DataFrame(columns=["port","year","month","month_index","w_from_q"])
    df_q = df_tons_pm.copy()
    df_q["quarter"] = df_q["month"].map(month_to_quarter)
    tons_q = df_q.groupby(["port","year","quarter"], as_index=False)["tons_p_m"].sum().rename(columns={"tons_p_m":"tons_p_q"})
    pq = df_teu_pq[df_teu_pq["port"].isin(CANON_PORTS)].copy()
    qmerge = tons_q.merge(pq, on=["port","year","quarter"], how="inner")
    qmerge["r_q"] = np.where(qmerge["teu_p_q"]>0, qmerge["tons_p_q"]/qmerge["teu_p_q"], np.nan)
    qmerge["r_q_w"] = winsorize_group(qmerge, "r_q", by=["port","year"], lower=0.01, upper=0.99)
    mu = qmerge.groupby(["port","year"])["r_q_w"].transform("mean")
    qmerge["w_p_q"] = np.where((mu==0) | (mu.isna()), 1.0, qmerge["r_q_w"]/mu).astype(float)
    qmerge["w_p_q"] = qmerge["w_p_q"].fillna(1.0)

    months_by_q = {"Q1":[1,2,3], "Q2":[4,5,6], "Q3":[7,8,9], "Q4":[10,11,12]}
    rows = []
    for _, r in qmerge.iterrows():
        wq = r.get("w_p_q", np.nan)
        wq_val = float(wq) if pd.notna(wq) else np.nan
        for m in months_by_q.get(r["quarter"], []):
            rows.append({"port":r["port"], "year":int(r["year"]), "month":int(m),
                         "month_index": int(r["year"]*100 + int(m)),
                         "w_from_q": wq_val})
    out = pd.DataFrame(rows)
    return out

def compute_pi_mixbase_port_month(df_l_tm: pd.DataFrame) -> pd.DataFrame:
    df_q = (
        df_l_tm.groupby(["port","terminal","year","quarter"], as_index=False)["teu_i_m"]
        .sum().rename(columns={"teu_i_m":"teu_i_q"})
    )
    df_q["sum_port_q"] = df_q.groupby(["port","year","quarter"])["teu_i_q"].transform("sum")
    df_q["s_i_p_q"] = np.where(df_q["sum_port_q"]>0, df_q["teu_i_q"]/df_q["sum_port_q"], 0.0)
    df_m = df_l_tm[["port","terminal","year","month","quarter","pi_teu_per_hour_i_y"]].copy()
    df_m = df_m.merge(df_q[["port","terminal","year","quarter","s_i_p_q"]],
                      on=["port","terminal","year","quarter"], how="left")
    df_m["contrib"] = df_m["s_i_p_q"].fillna(0.0) * pd.to_numeric(df_m["pi_teu_per_hour_i_y"], errors="coerce")
    df_pi = df_m.groupby(["port","year","month"], as_index=False)["contrib"].sum().rename(columns={"contrib":"pi_p_y_mixbase"})
    df_pi["month_index"] = df_pi["year"]*100 + df_pi["month"]
    return df_pi

# ---------------- build LP ----------------

def build_lp_port(df_w_final: pd.DataFrame, df_pi_p_m: pd.DataFrame, df_l_tm: pd.DataFrame, df_tons_pm: pd.DataFrame):
    df_lp = df_w_final.merge(df_pi_p_m[["port","year","month","pi_p_y_mixbase"]], on=["port","year","month"], how="left")
    df_lp["lp_port_month_mix"] = df_lp["w_final"] * df_lp["pi_p_y_mixbase"]
    diag = df_tons_pm[["port","year","month","month_index","teu_p_m","tons_p_m","tons_source"]]
    df_lp = df_lp.merge(diag, on=["port","year","month","month_index"], how="left")
    df_lp = df_lp[["port","year","month","month_index","teu_p_m","tons_p_m","w_final","pi_p_y_mixbase","lp_port_month_mix","tons_source"]].copy()
    df_L_port = df_l_tm.groupby(["port","year","month"], as_index=False)["l_hours_i_m"].sum().rename(columns={"l_hours_i_m":"l_port_m"})
    df_id = df_tons_pm[["port","year","month","teu_p_m"]].merge(df_L_port, on=["port","year","month"], how="left")
    df_id["lp_port_month_id"] = np.where(df_id["l_port_m"]>0, df_id["teu_p_m"]/df_id["l_port_m"], np.nan)
    df_id = df_id[["port","year","month","l_port_m","lp_port_month_id"]]
    df_lp_full = df_lp.merge(df_id, on=["port","year","month"], how="left")
    return df_lp_full, df_id

def build_lp_terminal(df_w_final: pd.DataFrame, df_l_tm: pd.DataFrame) -> pd.DataFrame:
    # Merge w_final to terminal records
    df = df_l_tm.merge(df_w_final[["port","year","month","w_final"]], on=["port","year","month"], how="left")
    # Safety: if any w_final missing, backfill from port-month mean (identical value per port-month)
    if df["w_final"].isna().any():
        pm = df[["port","year","month","w_final"]].drop_duplicates().groupby(["port","year","month"])["w_final"].transform("mean")
        df["w_final"] = df["w_final"].fillna(pm)

    # Opening month per terminal: first month with TEU_i_m>0 OR L_hours_i_m>0
    df_sorted = df.sort_values(["port","terminal","year","month"])
    df_sorted["has_activity"] = (pd.to_numeric(df_sorted["teu_i_m"], errors="coerce")>0) | (pd.to_numeric(df_sorted["l_hours_i_m"], errors="coerce")>0)
    df_sorted["open_flag"] = df_sorted.groupby(["port","terminal"])["has_activity"].transform(lambda s: s.cumsum()>0)
    # Compute LP regardless of TEU_i_m/L in post-opening months; keep NA strictly pre-opening
    df_sorted["lp_term_month_mixadjusted"] = pd.to_numeric(df_sorted["pi_teu_per_hour_i_y"], errors="coerce") * pd.to_numeric(df_sorted["w_final"], errors="coerce")
    df_sorted.loc[~df_sorted["open_flag"], "lp_term_month_mixadjusted"] = np.nan

    out = df_sorted[["port","terminal","year","month","month_index","quarter","open_flag","pi_teu_per_hour_i_y","w_final","teu_i_m","l_hours_i_m","lp_term_month_mixadjusted"]].copy()
    return out

def build_unified_panel(df_lp_port_full: pd.DataFrame, df_lp_term: pd.DataFrame) -> pd.DataFrame:
    port = df_lp_port_full.copy()
    port["level"] = "port"; port["terminal"] = pd.NA
    port["Pi"] = port["pi_p_y_mixbase"]
    port["L_hours"] = port["l_port_m"]
    port["LP_mix"] = port["lp_port_month_mix"]
    port["LP_id"] = port["lp_port_month_id"]
    port["quarter"] = "Q" + (((port["month"]-1)//3)+1).astype(str)
    port["TEU"] = port["teu_p_m"]; port["tons"] = port["tons_p_m"]
    port["w"] = port["w_final"]
    port_panel = port[["level","port","terminal","year","month","month_index","quarter","TEU","tons","w","Pi","L_hours","LP_mix","LP_id","tons_source"]]

    term = df_lp_term.copy()
    term["level"] = "terminal"
    term["Pi"] = term["pi_teu_per_hour_i_y"]
    term["L_hours"] = term["l_hours_i_m"]
    term["LP_mix"] = term["lp_term_month_mixadjusted"]
    term["LP_id"] = pd.NA
    term["TEU"] = term["teu_i_m"]; term["tons"] = pd.NA
    term["w"] = term["w_final"]; term["tons_source"] = pd.NA
    term_panel = term[["level","port","terminal","year","month","month_index","quarter","TEU","tons","w","Pi","L_hours","LP_mix","LP_id","tons_source"]]

    panel = pd.concat([port_panel, term_panel], ignore_index=True).sort_values(["level","port","terminal","year","month"]).reset_index(drop=True)
    return panel

# ---------------- QA ----------------

def run_qa(df_lp_port: pd.DataFrame, df_lp_term: pd.DataFrame, df_w_final: pd.DataFrame):
    rows = []
    def assert_unique(df, keys, name):
        c = df.duplicated(keys).sum()
        rows.append({"check":f"unique_keys_{name}", "result":"pass" if c==0 else "fail", "detail":f"duplicates={c} keys={keys}"})
    assert_unique(df_lp_port, ["port","year","month"], "lp_port")
    assert_unique(df_lp_term, ["port","terminal","year","month"], "lp_term")
    assert_unique(df_w_final, ["port","year","month"], "w_final")
    g = df_lp_port.groupby(["port","year"], as_index=False).agg(lp_mean=("lp_port_month_mix","mean"), pi_mean=("pi_p_y_mixbase","mean"))
    g["rel_err"] = np.abs(g["lp_mean"]-g["pi_mean"])/g["pi_mean"].replace(0,np.nan)
    for _, r in g.iterrows():
        rows.append({"check":"annual_preservation","port":r["port"],"year":int(r["year"]),
                     "lp_mean":r["lp_mean"],"pi_mean":r["pi_mean"],
                     "rel_err":r["rel_err"],"result":"pass" if (pd.isna(r["rel_err"]) or r["rel_err"]<=1e-6) else "warn"})
    return pd.DataFrame(rows)

def qa_terminals_coverage(df_lp_term: pd.DataFrame) -> pd.DataFrame:
    df = df_lp_term.copy()
    out = []
    for (p,t), g in df.groupby(["port","terminal"]):
        g2024 = g[g["year"]==2024]
        out.append({
            "port": p, "terminal": t,
            "first_obs": f"{int(g['year'].min())}-{int(g['month'].min()):02d}",
            "last_obs":  f"{int(g['year'].max())}-{int(g['month'].max()):02d}",
            "n_2024_rows": int(len(g2024)),
            "n_2024_lp_nonnull": int(g2024["lp_term_month_mixadjusted"].notna().sum())
        })
    return pd.DataFrame(out).sort_values(["port","terminal"])

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teu", default=str(TEU_DEFAULT))
    ap.add_argument("--tons", default=str(TONS_DEFAULT))
    ap.add_argument("--l-proxy", dest="l_proxy", default=str(L_PROXY_DEFAULT))
    ap.add_argument("--out-port", default=str(OUT_PORT_DEFAULT))
    ap.add_argument("--out-term", default=str(OUT_TERM_DEFAULT))
    ap.add_argument("--out-id", default=str(OUT_ID_DEFAULT))
    ap.add_argument("--out-panel", default=str(OUT_PANEL_DEFAULT))
    ap.add_argument("--out-qa", default=str(OUT_QA_DEFAULT))
    ap.add_argument("--out-qa-terms", default=str(OUT_QA_TERMS_DEFAULT))
    ap.add_argument("--out-meta", default=str(OUT_META_DEFAULT))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allocation-ports", nargs="*", default=ALLOCATION_PORTS_DEFAULT)
    args = ap.parse_args()

    if not os.path.exists(args.teu):
        alt = str(TEU_FALLBACK)
        if os.path.exists(alt):
            print(f"[info] TEU file not found at {args.teu}; using fallback {alt}")
            args.teu = alt

    teu_m, teu_q = load_teu_split(args.teu)
    tons_mixed = load_tons_mixed(args.tons)
    l_tm = load_l_proxy(args.l_proxy)

    # Canonicalize keys used in later merges
    for df in (teu_m, teu_q, tons_mixed, l_tm):
        if "port" in df.columns: df["port"] = df["port"].map(canon_port_name)

    tons_pm = build_tons_pm(tons_mixed, teu_m if not teu_m.empty else pd.DataFrame(columns=["port","year","month","teu_p_m"]), args.allocation_ports)

    w_m = compute_w_monthly(tons_pm)
    w_q_broadcast = compute_w_quarter_fallback(tons_pm, teu_q)

    w_final = (
        pd.merge(w_m, w_q_broadcast, on=["port","year","month","month_index"], how="outer")
          .assign(w_final=lambda d: d["w_p_m"].combine_first(d["w_from_q"]))
          .loc[:, ["port","year","month","month_index","w_final"]]
    )

    pi_p_m = compute_pi_mixbase_port_month(l_tm)

    lp_port_full, lp_id = build_lp_port(w_final, pi_p_m, l_tm, tons_pm)
    lp_term = build_lp_terminal(w_final, l_tm)
    panel = build_unified_panel(lp_port_full, lp_term)
    qa = run_qa(lp_port_full, lp_term, w_final)
    qa_terms = qa_terminals_coverage(lp_term)

    meta = {
        "script": str(HERE),
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "quarter_fallback": True,
        "inputs": {
            "teu":   {"path": args.teu, "sha256": sha256sum(args.teu) if os.path.exists(args.teu) else None},
            "tons":  {"path": args.tons, "sha256": sha256sum(args.tons) if os.path.exists(args.tons) else None},
            "l_proxy":{"path": args.l_proxy, "sha256": sha256sum(args.l_proxy) if os.path.exists(args.l_proxy) else None},
        },
        "row_counts": {
            "teu_monthly_rows": int(len(teu_m)),
            "teu_quarterly_rows": int(len(teu_q)),
            "tons_mixed_rows": int(len(tons_mixed)),
            "l_proxy_rows": int(len(l_tm)),
            "tons_pm_rows": int(len(tons_pm)),
            "w_monthly_rows": int(len(w_m)),
            "w_q_broadcast_rows": int(len(w_q_broadcast)),
            "w_final_rows": int(len(w_final)),
            "lp_port_rows": int(len(lp_port_full)),
            "lp_term_rows": int(len(lp_term)),
            "panel_rows": int(len(panel)),
            "qa_rows": int(len(qa)),
        }
    }

    def _write(path, df):
        write_tsv(df, path, force=args.force)

    _write(args.out_port, lp_port_full.sort_values(["port","year","month"]))
    _write(args.out_term, lp_term.sort_values(["port","terminal","year","month"]))
    _write(args.out_id, lp_id.sort_values(["port","year","month"]))
    _write(args.out_panel, panel)
    _write(args.out_qa, qa)
    _write(args.out_qa_terms, qa_terms)

    Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:")
    for p in [args.out_port, args.out_term, args.out_id, args.out_panel, args.out_qa, args.out_qa_terms, args.out_meta]:
        print("  ", p)

if __name__ == "__main__":
    main()
