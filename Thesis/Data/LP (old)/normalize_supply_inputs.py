#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize_supply_inputs.py  (v3 — strict, frequency-aware, MonthIndex-safe)

This version fixes TEU normalization once and for all by using a
STRICT schema contract for the TEU source and hard QA gates.

Key guarantees
--------------
- Uses only these TEU columns (case-insensitive, exact or word-boundary match):
  Port | Period | Freq | Year | MonthIndex | TEU (or TEU_thousands)
- Parses MonthIndex (YYYYMM) → (year, month) and computes month_index correctly.
- Routes by Freq only: Monthly → teu_port_month; Quarterly → teu_port_quarter.
- Derives quarter labels from month for quarterly rows; validates month ∈ {3,6,9,12}.
- Drops "All Ports" at ingest.
- De-leaks quarter totals from monthly at quarter-ends.
- Strict deduplication: monthly keeps MIN per (p,y,m); quarterly keeps MAX per (p,y,Qk).
- Hard QA gates; if any fail, writes _teu_errors.tsv and exits non-zero (no partial outputs).

Outputs (written to --out)
--------------------------
- tons_port_month.tsv        (port,year,month,month_index,tons_p_m,tons_source)
- tons_terminal_month.tsv    (port,terminal,year,month,month_index,tons_i_m)
- teu_port_month.tsv         (port,year,month,month_index,teu_p_m)
- teu_port_quarter.tsv       (port,year,quarter,teu_p_q)
- l_proxy.tsv                (port,terminal,year,month,month_index,quarter,l_hours_i_m,teu_i_m,pi_teu_per_hour_i_y,operating)
- _schema_map.json
- _coverage_report.tsv
- _normalizer_log.json
- (on error) _teu_errors.tsv
"""
import argparse, os, re, json, math, sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# ---------------- helpers ----------------

def _read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", engine="python")

def _write_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False)

def _quarter_from_month(m) -> Optional[str]:
    if pd.isna(m): return None
    try:
        q = (int(m)-1)//3 + 1
        return f"Q{q}"
    except Exception:
        return None

def _norm_port(s) -> Optional[str]:
    if s is None or (isinstance(s, float) and math.isnan(s)): return None
    s2 = str(s).replace("–","-").strip()
    low = s2.lower()
    if low.startswith("ashdod"): return "Ashdod"
    if low.startswith("haifa"): return "Haifa"
    if low.startswith("eilat"): return "Eilat"
    if low in {"all ports","all_ports","allports","all"}: return "All Ports"
    if "-" in s2:  # Port-Terminal style → port prefix
        pref = s2.split("-",1)[0].strip()
        return _norm_port(pref)
    return s2

# strict resolver: exact (case-insensitive) or whole-word boundary only

def _resolve_one(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if not candidates: return None
    lc = {c.lower(): c for c in df.columns}
    # exact first
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    # whole-word boundary regex
    for cand in candidates:
        pat = re.compile(rf"\b{re.escape(cand.lower())}\b")
        for c in df.columns:
            if pat.search(c.lower()):
                return c
    return None

# -------------- Tons --------------

def normalize_tons(tons_path: str):
    df = _read_tsv(tons_path)
    port_or_term = _resolve_one(df, ["PortOrTerminal","Port_or_Terminal","Port/Terminal","Port or Terminal","Port","Terminal","Location"]) or "PortOrTerminal"
    period_col  = _resolve_one(df, ["Period","Month-Year","Date","MonthYear","Year-Month","Year_Month","YM"]) or "Period"
    tons_col    = _resolve_one(df, ["Tons_1000","Tons_thousands","Tons_K","Ktons","1000_Tons","Thousand_Tons","Tons","Value","Amount"]) or "Tons"

    # parse period
    def _ym_from_token(x):
        if pd.isna(x): return (None, None)
        s = str(x).strip()
        for fmt in ("%m-%Y","%Y-%m","%m/%Y","%Y/%m","%b-%Y","%Y%b"):
            try:
                dt = pd.to_datetime(s, format=fmt, errors="raise")
                return int(dt.year), int(dt.month)
            except Exception:
                pass
        if s.isdigit() and len(s)==6:
            y = int(s[:4]); m = int(s[4:])
            return (y,m)
        m = re.search(r"(^|[^\d])(1[0-2]|0?[1-9])($|[^\d])", s)
        y = re.search(r"(19|20)\d{2}", s)
        if m and y:
            return (int(y.group(0)), int(m.group(2)))
        return (None,None)

    years, months = zip(*[ _ym_from_token(v) for v in df[period_col].tolist() ])
    tons_raw = pd.to_numeric(df[tons_col], errors="coerce")
    scale = 1000.0 if any(k in tons_col.lower() for k in ["1000","k","thousand"]) else 1.0
    tons = tons_raw * scale

    pot = df[port_or_term].astype(str).str.strip().replace({"": np.nan})
    out = pd.DataFrame({
        "port_or_terminal": pot,
        "port": pot.apply(_norm_port),
        "terminal": np.where(pot.str.contains("-", regex=False), pot, pd.NA),
        "year": pd.Series(years, dtype="Int64"),
        "month": pd.Series(months, dtype="Int64"),
        "tons": tons
    })
    out = out[(out["year"].notna()) & (out["month"].notna())].copy()

    # drop All Ports
    out = out[out["port"].str.lower() != "all ports"].copy()

    # precedence: port_total > sum_terminals
    is_port_total = out["terminal"].isna()
    tot = out.loc[is_port_total, ["port","year","month","tons"]].rename(columns={"tons":"tons_from_port_total"})
    term= out.loc[~is_port_total, ["port","terminal","year","month","tons"]].rename(columns={"tons":"tons_i_m"})
    term_sum = (term.groupby(["port","year","month"], dropna=False)["tons_i_m"].sum(min_count=1).reset_index()
                .rename(columns={"tons_i_m":"tons_from_sum_terminals"}))
    key = pd.concat([tot[["port","year","month"]], term_sum[["port","year","month"]]], ignore_index=True).drop_duplicates()
    merged = key.merge(tot, on=["port","year","month"], how="left").merge(term_sum, on=["port","year","month"], how="left")
    merged["tons_p_m"] = merged["tons_from_port_total"].combine_first(merged["tons_from_sum_terminals"])
    merged["tons_source"] = np.where(merged["tons_from_port_total"].notna(), "port_total",
                               np.where(merged["tons_from_sum_terminals"].notna(), "sum_terminals", "no_source"))

    tons_port_month = merged[["port","year","month","tons_p_m","tons_source"]].copy()
    tons_port_month["month_index"] = (tons_port_month["year"].astype("Int64")*12 + tons_port_month["month"].astype("Int64")).astype("Int64")
    tons_port_month = tons_port_month.drop_duplicates(["port","year","month"]).sort_values(["port","year","month"]) 

    tons_terminal_month = term.copy()
    tons_terminal_month["month_index"] = (tons_terminal_month["year"].astype("Int64")*12 + tons_terminal_month["month"].astype("Int64")).astype("Int64")

    schema = {"port_or_terminal": port_or_term, "period": period_col, "tons_raw": tons_col}
    return tons_port_month, tons_terminal_month, schema

# -------------- TEU (STRICT) --------------

def normalize_teu_strict(teu_path: str, out_dir: str):
    raw = _read_tsv(teu_path)

    # required columns per contract
    port_col   = _resolve_one(raw, ["Port"]) or "Port"
    period_col = _resolve_one(raw, ["Period","Month-Year"])  # optional
    freq_col   = _resolve_one(raw, ["Freq","Frequency"]) or "Freq"
    year_col   = _resolve_one(raw, ["Year"])  # optional, used for sanity
    ym_col     = _resolve_one(raw, ["MonthIndex","YYYYMM"])  # preferred
    teu_col    = _resolve_one(raw, ["TEU"]) or _resolve_one(raw, ["TEU_thousands","TEU_thousand"]) or "TEU"

    missing = []
    if port_col is None: missing.append("Port")
    if freq_col is None: missing.append("Freq/Frequency")
    if ym_col is None and period_col is None: missing.append("MonthIndex or Period")
    if teu_col is None: missing.append("TEU or TEU_thousands")
    if missing:
        raise ValueError(f"[TEU] Missing required columns: {', '.join(missing)}")

    df = raw.copy()
    # port + drop All Ports
    df["port"] = df[port_col].astype(str).str.strip().replace({"": np.nan}).apply(_norm_port)
    df = df[df["port"].str.lower() != "all ports"].copy()

    # value selection
    scale = 1.0
    if teu_col.lower().startswith("teu_thousand"): scale = 1000.0
    df["teu_val"] = pd.to_numeric(df[teu_col], errors="coerce") * scale

    # time parsing (prefer MonthIndex)
    def parse_ym_token(s):
        if pd.isna(s): return (None, None)
        z = str(s).strip()
        m = re.match(r"^(\d{4})(\d{2})$", z)
        if not m: return (None, None)
        y = int(m.group(1)); mo = int(m.group(2))
        return (y, mo)

    if ym_col is not None:
        ymo = [parse_ym_token(v) for v in df[ym_col].tolist()]
        year, month = zip(*ymo) if len(ymo) else ([],[])
        df["year"]  = pd.Series(year, dtype="Int64")
        df["month"] = pd.Series(month, dtype="Int64")
    elif period_col is not None:
        def _ym_from_period(x):
            if pd.isna(x): return (None,None)
            s = str(x).strip()
            for fmt in ("%m-%Y","%Y-%m","%m/%Y","%Y/%m","%b-%Y","%Y%b"):
                try:
                    dt = pd.to_datetime(s, format=fmt, errors="raise")
                    return int(dt.year), int(dt.month)
                except Exception:
                    pass
            return (None,None)
        pr = [ _ym_from_period(v) for v in df[period_col].tolist() ]
        year, month = zip(*pr) if len(pr) else ([],[])
        df["year"]  = pd.Series(year, dtype="Int64")
        df["month"] = pd.Series(month, dtype="Int64")

    # sanity: month must be 1..12; if not, treat as bad
    bad_month = df["month"].notna() & ~df["month"].between(1,12)
    if bad_month.any():
        # most likely someone fed MonthIndex into month; error out explicitly
        bad = df.loc[bad_month, ["port","year","month"]].head(10)
        _write_tsv(bad, os.path.join(out_dir, "_teu_errors.tsv"))
        raise SystemExit("[TEU] Invalid month values detected (not 1..12). Wrote examples to _teu_errors.tsv")

    # month_index
    df["month_index"] = (df["year"].astype("Int64")*12 + df["month"].astype("Int64")).astype("Int64")

    # frequency routing (no inference)
    f = df[freq_col].astype(str).str.strip().str.lower()
    is_m = f.str.startswith("m")
    is_q = f.str.startswith("q")

    # monthly
    mdf = df[is_m].copy()
    mdf = mdf[(mdf["year"].notna()) & (mdf["month"].notna())]
    teu_m = mdf[["port","year","month","month_index"]].copy()
    teu_m["teu_p_m"] = pd.to_numeric(mdf["teu_val"], errors="coerce")

    # quarterly
    qdf = df[is_q].copy()
    # derive quarter from month; require q-end months
    qdf = qdf[qdf["year"].notna() & qdf["month"].isin([3,6,9,12])].copy()
    qdf["quarter"] = qdf["month"].map({3:"Q1",6:"Q2",9:"Q3",12:"Q4"})
    teu_q = qdf[["port","year","quarter"]].copy()
    teu_q["teu_p_q"] = pd.to_numeric(qdf["teu_val"], errors="coerce")

    # strict dedup rules
    if not teu_m.empty:
        teu_m = (teu_m.sort_values(["port","year","month","teu_p_m"], ascending=[True,True,True,True])
                      .drop_duplicates(["port","year","month"], keep="first"))
    if not teu_q.empty:
        teu_q = (teu_q.sort_values(["port","year","quarter","teu_p_q"], ascending=[True,True,True,False])
                      .drop_duplicates(["port","year","quarter"], keep="first"))

    # de-leak monthly at quarter ends where quarterly total exists
    if not teu_m.empty and not teu_q.empty:
        qmap = {"Q1":3, "Q2":6, "Q3":9, "Q4":12}
        qq = teu_q.copy(); qq["m_qend"] = qq["quarter"].map(qmap)
        merged = teu_m.merge(qq, left_on=["port","year","month"], right_on=["port","year","m_qend"], how="left")
        leak = merged["teu_p_q"].notna() & (merged["teu_p_m"] >= merged["teu_p_q"]*0.999)
        drop = merged.loc[leak, ["port","year","month"]].drop_duplicates()
        if not drop.empty:
            teu_m = teu_m.merge(drop, on=["port","year","month"], how="left", indicator=True)
            teu_m = teu_m[teu_m["_merge"]=="left_only"].drop(columns=["_merge"])

    # QA gates
    errs = []
    # 1) unique keys
    if teu_m.duplicated(["port","year","month"]).any(): errs.append("Duplicate (port,year,month) in monthly TEU")
    if teu_q.duplicated(["port","year","quarter"]).any(): errs.append("Duplicate (port,year,quarter) in quarterly TEU")
    # 2) month range and month_index consistency
    if (not teu_m.empty) and (not teu_m["month"].between(1,12).all()): errs.append("Monthly TEU has month outside 1..12")
    if not teu_m.empty:
        check_mi = (teu_m["year"].astype("Int64")*12 + teu_m["month"].astype("Int64")).astype("Int64")
        if not (check_mi.equals(teu_m["month_index"].astype("Int64"))): errs.append("month_index mismatch in monthly TEU")
    # 3) quarter labels set
    if (not teu_q.empty) and (set(teu_q["quarter"]).difference({"Q1","Q2","Q3","Q4"})): errs.append("Invalid quarter labels in quarterly TEU")
    # 4) coverage sanity (at least monthly or full quarterly)
    if not teu_m.empty or not teu_q.empty:
        ports = sorted(set(pd.concat([teu_m["port"], teu_q["port"]]).dropna().unique()))
        for p in ports:
            years = sorted(set(pd.concat([teu_m.loc[teu_m["port"]==p, "year"], teu_q.loc[teu_q["port"]==p, "year"]]).dropna().unique()))
            for y in years:
                mN = int(teu_m[(teu_m.port==p)&(teu_m.year==y)].shape[0])
                qN = int(teu_q[(teu_q.port==p)&(teu_q.year==y)].shape[0])
                if mN==0 and qN not in {0,4}: errs.append(f"Coverage inconsistent for {p}-{y}: monthly=0 & quarterly={qN}")

    if errs:
        # write errors and abort
        _write_tsv(pd.DataFrame({"error": errs}), os.path.join(out_dir, "_teu_errors.tsv"))
        raise SystemExit("; ".join(errs))

    # final column order
    teu_m = teu_m[["port","year","month","month_index","teu_p_m"]].sort_values(["port","year","month"]).reset_index(drop=True)
    teu_q = teu_q[["port","year","quarter","teu_p_q"]].sort_values(["port","year","quarter"]).reset_index(drop=True)

    schema = {"port": port_col, "period": period_col or "", "freq": freq_col, "year": year_col or "",
              "monthindex": ym_col or "", "teu_value": teu_col}
    return teu_m, teu_q, schema

# -------------- L_Proxy --------------

def normalize_lproxy(lproxy_path: str):
    df = _read_tsv(lproxy_path)
    def pick(cands: List[str]): return _resolve_one(df, cands)
    port_col = pick(["Port","port"]) or "port"
    term_col = pick(["Terminal","terminal","Terminal_Name"]) or "terminal"
    year_col = pick(["Year","year"]) or "year"
    month_col= pick(["Month","month","MonthIndex"]) or "month"
    hours_col= pick(["l_hours_i_m","l_hours","hours_i_m","hours","Labor_Hours"]) or "l_hours_i_m"
    teu_col  = pick(["teu_i_m","TEU_i_m","TEU_terminal","teu"]) or "teu_i_m"
    pi_col   = pick(["pi_teu_per_hour_i_y","pi","pi_i_y"]) or "pi_teu_per_hour_i_y"
    oper_col = pick(["operating","is_operating","open"])  # optional

    g = pd.DataFrame({
        "port": df[port_col].astype(str).str.strip().replace({"": np.nan}).apply(_norm_port),
        "terminal": df[term_col].astype(str).str.strip().replace({"": np.nan}),
        "year": pd.to_numeric(df[year_col], errors="coerce").astype("Int64"),
        "month": pd.to_numeric(df[month_col], errors="coerce").astype("Int64"),
        "l_hours_i_m": pd.to_numeric(df[hours_col], errors="coerce") if hours_col else np.nan,
        "teu_i_m": pd.to_numeric(df[teu_col], errors="coerce") if teu_col else np.nan,
        "pi_teu_per_hour_i_y": pd.to_numeric(df[pi_col], errors="coerce") if pi_col else np.nan,
        "operating": (df[oper_col].astype(str).str.strip() if oper_col else pd.NA)
    })
    g = g[(g["year"].notna()) & (g["month"].notna())].copy()
    g["month_index"] = (g["year"].astype("Int64")*12 + g["month"].astype("Int64")).astype("Int64")
    g["quarter"] = g["month"].apply(_quarter_from_month)

    schema = {"port": port_col, "terminal": term_col, "year": year_col, "month": month_col,
              "l_hours_i_m": hours_col or "", "teu_i_m": teu_col or "", "pi_teu_per_hour_i_y": pi_col or "", "operating": oper_col or ""}
    return g, schema

# -------------- Coverage --------------

def build_coverage(tons_pm, teu_pm, teu_pq, lpr):
    ports = sorted(set(pd.concat([tons_pm["port"], lpr["port"]]).dropna().unique()))
    rows = []
    for p in ports:
        yrs = sorted(set(pd.concat([
            tons_pm.loc[tons_pm["port"]==p, "year"],
            lpr.loc[lpr["port"]==p, "year"],
            teu_pm.loc[teu_pm["port"]==p, "year"] if not teu_pm.empty else pd.Series(dtype="Int64"),
            teu_pq.loc[teu_pq["port"]==p, "year"] if not teu_pq.empty else pd.Series(dtype="Int64"),
        ]).dropna().unique()))
        for y in yrs:
            tN = int(tons_pm[(tons_pm.port==p)&(tons_pm.year==y)].shape[0])
            mN = int(teu_pm[(teu_pm.port==p)&(teu_pm.year==y)].shape[0]) if not teu_pm.empty else 0
            qN = int(teu_pq[(teu_pq.port==p)&(teu_pq.year==y)].shape[0]) if not teu_pq.empty else 0
            lN = int(lpr[(lpr.port==p)&(lpr.year==y)].shape[0])
            rows.append({"port":p,"year":int(y),"tons_pm_n":tN,"teu_pm_n":mN,"teu_pq_n":qN,"lproxy_n":lN,"note":"ok"})
    return pd.DataFrame(rows)

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tons", required=True)
    ap.add_argument("--teu", required=True)
    ap.add_argument("--lproxy", required=True)
    ap.add_argument("--out", default="Data/LP/_normalized")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Normalize Tons
    tons_pm, tons_term_m, cmap_tons = normalize_tons(args.tons)

    # Normalize TEU (strict)
    teu_pm, teu_pq, cmap_teu = normalize_teu_strict(args.teu, args.out)

    # Normalize L_Proxy
    l_proxy, cmap_lpr = normalize_lproxy(args.lproxy)

    # Final writes
    _write_tsv(tons_pm,     os.path.join(args.out, "tons_port_month.tsv"))
    _write_tsv(tons_term_m, os.path.join(args.out, "tons_terminal_month.tsv"))
    _write_tsv(teu_pm,      os.path.join(args.out, "teu_port_month.tsv"))
    _write_tsv(teu_pq,      os.path.join(args.out, "teu_port_quarter.tsv"))
    _write_tsv(l_proxy,     os.path.join(args.out, "l_proxy.tsv"))

    # schema + coverage + log
    schema_map = {"tons": cmap_tons, "teu": cmap_teu, "l_proxy": cmap_lpr}
    with open(os.path.join(args.out, "_schema_map.json"), "w", encoding="utf-8") as f:
        json.dump(schema_map, f, indent=2, ensure_ascii=False)

    coverage = build_coverage(tons_pm, teu_pm, teu_pq, l_proxy)
    _write_tsv(coverage, os.path.join(args.out, "_coverage_report.tsv"))

    log = {
        "notes": "TEU normalized with strict schema and QA gates; MonthIndex parsed to (year,month)."
    }
    with open(os.path.join(args.out, "_normalizer_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print(f"[normalize] wrote normalized files to: {args.out}")

if __name__ == "__main__":
    main()
