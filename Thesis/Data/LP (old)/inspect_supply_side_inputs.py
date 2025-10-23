
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import re
from typing import Optional

def _pick(df: pd.DataFrame, names, contains_ok=True) -> Optional[str]:
    # exact first
    for cand in names:
        for c in df.columns:
            if c.lower() == cand.lower():
                return c
    if contains_ok:
        for cand in names:
            for c in df.columns:
                if cand.lower() in c.lower():
                    return c
    return None

def _norm_port(x: str) -> str:
    if pd.isna(x): return x
    s = str(x).replace("–","-").strip()
    low = s.lower()
    if low.startswith("ashdod"): return "Ashdod"
    if low.startswith("haifa"):  return "Haifa"
    if low.startswith("eilat"):  return "Eilat"
    if low in {"all ports","all_ports","allports","all"}: return "All Ports"
    return s

def _split_port_terminal(val: str):
    if pd.isna(val): return (val, None)
    s = str(val).strip()
    if s.lower() in {"all ports","all_ports","allports","all"}:
        return ("All Ports", None)
    if "-" in s:
        left, right = s.split("-", 1)
        return (_norm_port(left.strip()), right.strip())
    # default: treat as port total
    return (_norm_port(s), None)

def _parse_month_year_col(series: pd.Series) -> pd.DataFrame:
    # Accept formats like '01-2008', '2008-01', 'Jan-2008', '2008/01', etc.
    # Also accept 'YYYYQ#' as quarter (we'll expand to months with NaNs for monthly TEU path).
    s = series.astype(str).str.strip()
    # Try pandas auto
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    out = pd.DataFrame({"year": dt.dt.year, "month": dt.dt.month})
    # If still many NAs, try manual regex for mm-YYYY or YYYY-mm
    mask = out["year"].isna() | out["month"].isna()
    if mask.mean() > 0.5:
        # mm-YYYY
        m = s.str.extract(r'(?P<m>\d{1,2})[-/](?P<y>\d{4})')
        y = pd.to_numeric(m.get("y"), errors="coerce")
        mm = pd.to_numeric(m.get("m"), errors="coerce")
        out.loc[mask & y.notna() & mm.notna(), "year"] = y
        out.loc[mask & y.notna() & mm.notna(), "month"] = mm
    # Try YYYY-mm
    mask = out["year"].isna() | out["month"].isna()
    if mask.mean() > 0.5:
        m2 = s.str.extract(r'(?P<y>\d{4})[-/](?P<m>\d{1,2})')
        y2 = pd.to_numeric(m2.get("y"), errors="coerce")
        mm2 = pd.to_numeric(m2.get("m"), errors="coerce")
        out.loc[mask & y2.notna() & mm2.notna(), "year"] = y2
        out.loc[mask & y2.notna() & mm2.notna(), "month"] = mm2
    return out

def load_tons(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", engine="python")
    port_col = _pick(df, ["port","port_name","portorterminal","location"])
    term_col = _pick(df, ["terminal"])
    tons_col = _pick(df, ["tons","tons_k","1000_tons","thousand_tons","ktons","value","amount"], contains_ok=True)
    year_col = _pick(df, ["year","yr"])
    month_col = _pick(df, ["month","mo"])
    monthyear_col = _pick(df, ["month-year","month_year","period","date"])

    if tons_col is None:
        raise SystemExit("Tons file: couldn't find a numeric column like 'tons'/'tons_k'.")

    if (year_col is None or month_col is None) and (monthyear_col is None):
        raise SystemExit("Tons file: need either (year & month) OR a 'Month-Year' column.")

    # Build basic frame
    out = pd.DataFrame()
    # Resolve port & terminal
    if port_col is not None and term_col is not None:
        out["port"] = df[port_col].map(_norm_port)
        out["terminal"] = df[term_col].astype(str).str.strip()
    elif port_col is not None:
        # might be "PortOrTerminal" mixed
        split_pt = df[port_col].apply(_split_port_terminal)
        out["port"] = split_pt.apply(lambda t: t[0])
        out["terminal"] = split_pt.apply(lambda t: t[1])
    else:
        raise SystemExit("Tons file: couldn't find a port or PortOrTerminal column.")

    # Resolve year-month
    if year_col is not None and month_col is not None:
        out["year"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
        out["month"] = pd.to_numeric(df[month_col], errors="coerce").astype("Int64")
    else:
        ym = _parse_month_year_col(df[monthyear_col])
        out["year"] = ym["year"].astype("Int64")
        out["month"] = ym["month"].astype("Int64")

    # Tons scale
    tons_raw = pd.to_numeric(df[tons_col], errors="coerce")
    if "1000" in tons_col.lower() or "k" in tons_col.lower() or "thousand" in tons_col.lower():
        out["tons"] = tons_raw * 1000.0
    else:
        out["tons"] = tons_raw
    # keep only ports (exclude All Ports for port-month coverage)
    out = out[out["port"].notna() & (out["port"]!="All Ports")]
    return out

def _parse_quarter_field(q) -> Optional[int]:
    if pd.isna(q): return None
    s = str(q).upper().replace(" ", "")
    m = re.search(r"Q([1-4])", s)
    if m: return int(m.group(1))
    if s.isdigit():
        qi = int(s)
        if 1 <= qi <= 4:
            return qi
    return None

def load_teu(path: str):
    df = pd.read_csv(path, sep="\t", engine="python")
    port_col = _pick(df, ["port","port_name","location"])
    year_col = _pick(df, ["year","yr"])
    month_col = _pick(df, ["month","mo"])
    quarter_col = _pick(df, ["quarter","qtr","q"])
    period_col = _pick(df, ["period","date","year_quarter","yr_qtr","yyyyq","yyyq","yyyyqq"])
    teu_col = _pick(df, ["teu","value","count","qty"])

    if port_col is None or year_col is None or teu_col is None:
        raise SystemExit("TEU file: need 'port', 'year', and a TEU numeric column.")

    base = pd.DataFrame({"port": df[port_col].map(_norm_port),
                         "year": pd.to_numeric(df[year_col], errors="coerce").astype("Int64")})
    if month_col is not None and df[month_col].notna().any():
        m = base.copy()
        m["month"] = pd.to_numeric(df[month_col], errors="coerce").astype("Int64")
        m["teu"] = pd.to_numeric(df[teu_col], errors="coerce")
        teu_m = m.dropna(subset=["port","year","month"])
    else:
        teu_m = pd.DataFrame(columns=["port","year","month","teu"])

    if quarter_col is not None and df[quarter_col].notna().any():
        q = base.copy()
        qnum = df[quarter_col].apply(_parse_quarter_field)
        q["quarter"] = qnum.map({1:"Q1",2:"Q2",3:"Q3",4:"Q4"})
        q["teu"] = pd.to_numeric(df[teu_col], errors="coerce")
        teu_q = q.dropna(subset=["port","year","quarter"])
    elif period_col is not None and df[period_col].notna().any():
        q = base.copy()
        qnum = df[period_col].apply(_parse_quarter_field)
        q["quarter"] = qnum.map({1:"Q1",2:"Q2",3:"Q3",4:"Q4"})
        q["teu"] = pd.to_numeric(df[teu_col], errors="coerce")
        teu_q = q.dropna(subset=["port","year","quarter"])
    else:
        teu_q = pd.DataFrame(columns=["port","year","quarter","teu"])

    return teu_m, teu_q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tons", required=True)
    ap.add_argument("--teu", required=True)
    ap.add_argument("--out", default="Data/LP/_supply_coverage_report.tsv")
    args = ap.parse_args()

    tons = load_tons(args.tons)
    teu_m, teu_q = load_teu(args.teu)

    # Coverage by port-year
    cov_tons = (tons.dropna(subset=["year","month"])
                     .groupby(["port","year"]).agg(months_with_tons=("month", lambda s: s.dropna().nunique()))
                     .reset_index())
    cov_teu_m = (teu_m.dropna(subset=["year","month"])
                      .groupby(["port","year"]).agg(months_with_teu=("month", lambda s: s.dropna().nunique()))
                      .reset_index())
    cov_teu_q = (teu_q.dropna(subset=["year","quarter"])
                      .groupby(["port","year"]).agg(quarters_with_teu=("quarter","nunique"))
                      .reset_index())

    # Merge
    ports_years = pd.concat([cov_tons[["port","year"]], cov_teu_m[["port","year"]], cov_teu_q[["port","year"]]]).drop_duplicates()
    rep = (ports_years.merge(cov_tons, on=["port","year"], how="left")
                    .merge(cov_teu_m, on=["port","year"], how="left")
                    .merge(cov_teu_q, on=["port","year"], how="left"))
    rep["months_with_tons"] = rep["months_with_tons"].fillna(0).astype(int)
    rep["months_with_teu"] = rep["months_with_teu"].fillna(0).astype(int)
    rep["quarters_with_teu"] = rep["quarters_with_teu"].fillna(0).astype(int)
    rep["has_monthly_teu"] = rep["months_with_teu"] > 0
    rep["has_quarterly_teu"] = rep["quarters_with_teu"] > 0
    rep["note"] = np.where(~rep["has_monthly_teu"] & rep["has_quarterly_teu"],
                           "quarterly TEU fallback only",
                           np.where(rep["has_monthly_teu"], "monthly TEU available", "no TEU for this year"))

    # Save
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rep.sort_values(["port","year"]).to_csv(out_path, sep="\t", index=False)

    # Print compact summary
    print("[Supply coverage written]", out_path)
    for p, g in rep.groupby("port", dropna=True):
        g2 = g.sort_values("year")
        yrs = ", ".join(f"{int(y)}(tons={int(mt)}, mTEU={int(mm)}, qTEU={int(qq)})"
                        for y, mt, mm, qq in zip(g2["year"], g2["months_with_tons"], g2["months_with_teu"], g2["quarters_with_teu"]))
        print(f"  {p}: {yrs}")

if __name__ == "__main__":
    import os
    main()
