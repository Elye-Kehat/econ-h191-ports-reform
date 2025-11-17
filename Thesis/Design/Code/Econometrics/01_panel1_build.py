#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine LP sources to a single port×quarter panel in *levels* (LP).

Priority:
  1) LP_Panel: port×quarter (direct)
  2) LP_Panel: terminal×quarter -> port×quarter (geom-mean, i.e., mean of ln(LP) then exp)
  3) pre-panel (panel_port_quarter.csv): convert ln(LP) Y -> LP if needed

Monthly rows are dropped. No trimming here; do that in the econometrics step.
"""

from __future__ import annotations
import sys, json, re
from pathlib import Path
from typing import Optional, Iterable, Tuple
import numpy as np
import pandas as pd

# ---------------- utils ----------------
def choose_col(df: pd.DataFrame, cands: Iterable[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

def safe_mkdir(p: Path): p.mkdir(parents=True, exist_ok=True)

_QSTR_PATTERNS = [
    re.compile(r"^\s*(?P<y>\d{4})\s*[-\s/]*Q\s*(?P<q>[1-4])\s*$", re.IGNORECASE),
    re.compile(r"^\s*Q\s*(?P<q>[1-4])\s*[-\s/]*(?P<y>\d{4})\s*$", re.IGNORECASE),
]

def parse_qstr(s: str) -> Optional[Tuple[int,int]]:
    if s is None or (isinstance(s, float) and np.isnan(s)): return None
    raw = str(s).strip()
    for pat in _QSTR_PATTERNS:
        m = pat.match(raw)
        if m:
            y, q = int(m.group("y")), int(m.group("q"))
            if 1 <= q <= 4: return (y, q)
    compact = raw.upper().replace(" ", "").replace("-", "").replace("/", "")
    if "Q" in compact:
        try:
            y, q = compact.split("Q", 1)
            y, q = int(y), int(q)
            if 1 <= q <= 4: return (y, q)
        except Exception:
            return None
    return None

def build_qtuple(df: pd.DataFrame,
                 month_col: Optional[str],
                 year_col: Optional[str],
                 quart_col: Optional[str],
                 qstr_like: Iterable[str]) -> pd.Series:
    q = pd.Series([None]*len(df), index=df.index, dtype="object")
    # month->quarter
    if month_col is not None:
        if year_col is not None:
            y = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
            m = pd.to_numeric(df[month_col], errors="coerce")
            if pd.api.types.is_numeric_dtype(m):
                ts = pd.to_datetime(y.astype(str) + "-" + m.astype("Int64").astype(str) + "-01", errors="coerce")
            else:
                ts = pd.to_datetime(df[month_col], errors="coerce")
        else:
            ts = pd.to_datetime(df[month_col], errors="coerce")
        idx = ts.notna()
        if idx.any():
            q.loc[idx] = [(int(t.year), ((int(t.month)-1)//3)+1) for t in ts[idx]]
    # q-string
    if q.isna().any():
        for c in qstr_like:
            if c in df.columns:
                parsed = df[c].map(parse_qstr)
                fill = q.isna() & parsed.notna()
                if fill.any(): q.loc[fill] = parsed.loc[fill]
    # numeric y+q
    if q.isna().any() and (year_col is not None and quart_col is not None):
        y = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
        r = pd.to_numeric(df[quart_col], errors="coerce").astype("Int64")
        fill = q.isna() & y.notna() & r.notna()
        if fill.any():
            q.loc[fill] = [(int(yy), int(rr)) for yy, rr in zip(y[fill], r[fill])]
    return q

def qtuple_to_str(t: Tuple[int,int]) -> str: return f"{int(t[0])}Q{int(t[1])}"

# ---------------- main ----------------
def main():
    # Paths (no YAML needed)
    here = Path(__file__).resolve()
    thesis_root = here.parents[3]  # Econometrics -> Code -> Design -> Thesis
    p_lp = thesis_root / "Data" / "LP" / "LP_Panel.tsv"
    p_pre = thesis_root / "Design" / "Output Data" / "panel_port_quarter.csv"

    out_dir = thesis_root / "Design" / "Output Data" 
    safe_mkdir(out_dir)
    out_csv  = out_dir / "01_panel_port_quarter_full.csv"
    out_meta = out_dir / "01__meta_panel_port_quarter_full.json"


    print(f"[combine] LP_Panel: {p_lp}")
    print(f"[combine] pre-panel: {p_pre}")
    print(f"[combine] out: {out_csv}")

    if not p_lp.exists(): raise FileNotFoundError(f"Missing {p_lp}")
    if not p_pre.exists(): print("[combine] WARNING: pre-panel not found; proceeding without it.")

    # ---- Read LP_Panel (levels) ----
    LP = pd.read_csv(p_lp, sep="\t")
    port = choose_col(LP, ["port","Port","PORT"])
    level = choose_col(LP, ["level","Level","LEVEL","panel_level"])
    freq  = choose_col(LP, ["freq","Freq","FREQ"])
    term  = choose_col(LP, ["terminal","Terminal","TERMINAL"])
    year  = choose_col(LP, ["year","Year","YEAR","y"])
    quarter = choose_col(LP, ["quarter","Quarter","QUARTER","q","Q"])
    month = choose_col(LP, ["month","Month","MONTH","date","Date","m","mon"])
    qstr_like = [c for c in LP.columns if re.search(r"(qtr|quarter|qstring|q_str|qstr)", c, re.IGNORECASE)]
    lp_col = choose_col(LP, ["LP","LP_mix","lp_mix","Y","y"])
    lnlp_col = choose_col(LP, ["ln_LP","ln_lp_mix","ln_lp","lnY","ln_y"])

    if port is None: raise ValueError("LP_Panel missing 'port'")
    LP[port] = LP[port].astype(str).str.strip()
    LP = LP[LP[port].isin(["Haifa","Ashdod"])].copy()

    # robust terminal/port level
    lvl = LP[level].astype(str).str.lower().str.strip() if level else pd.Series("", index=LP.index)
    has_term_col = term is not None
    term_nonnull = LP[term].notna() if has_term_col else pd.Series(False, index=LP.index)
    is_terminal = (lvl.str.contains("terminal")) | term_nonnull
    is_portlvl  = ~is_terminal

    # build quarter tuples
    LP["_qtuple"] = build_qtuple(LP, month, year, quarter, qstr_like)

    # outcome in LEVELS
    if lp_col:
        LP["_LP"] = pd.to_numeric(LP[lp_col], errors="coerce")
    elif lnlp_col:
        LP["_LP"] = np.exp(pd.to_numeric(LP[lnlp_col], errors="coerce"))
    else:
        raise ValueError("Neither LP nor ln(LP) present in LP_Panel.")

    # Monthly/quarterly detection (for info only)
    is_quarterly = pd.Series(False, index=LP.index)
    if freq:
        f = LP[freq].astype(str).str.lower()
        is_quarterly = f.str.contains("quart") | f.str.contains("quarter") | f.str.contains("qtr")

    # (A) LP_Panel port×quarter direct
    A = (LP.loc[is_portlvl & LP["_qtuple"].notna()]
           .dropna(subset=["_LP","_qtuple"])
           .groupby([port,"_qtuple"], as_index=False)["_LP"].mean())
    A["_source"] = "LP_panel_port_quarter"

    # (B) LP_Panel terminal×quarter -> port×quarter (geom-mean)
    Braw = LP.loc[is_terminal & LP["_qtuple"].notna()].dropna(subset=["_LP","_qtuple"]).copy()
    if not Braw.empty:
        # geometric mean across terminals: mean of ln(LP) then exp
        B = (Braw.assign(_ln=lambda d: np.log(d["_LP"]))
                  .groupby([port,"_qtuple"], as_index=False)["_ln"].mean()
                  .rename(columns={"_ln":"_LP"}))
        B["_LP"] = np.exp(B["_LP"])
        B["_source"] = "LP_panel_termQ_to_portQ"
    else:
        B = pd.DataFrame(columns=[port,"_qtuple","_LP","_source"])

    # ---- Read pre-panel (may have ln(LP) as Y) ----
    if p_pre.exists():
        PRE = pd.read_csv(p_pre)
        pre_port = choose_col(PRE, ["port"])
        pre_qtr  = choose_col(PRE, ["qtr"])
        pre_Y    = choose_col(PRE, ["Y","lnY","ln_LP","ln_lp"])
        pre_LP   = choose_col(PRE, ["LP"])
        if pre_qtr is None: raise ValueError("pre-panel missing 'qtr'")

        PRE["_qtuple"] = PRE[pre_qtr].map(parse_qstr)
        if pre_LP:
            PRE["_LP"] = pd.to_numeric(PRE[pre_LP], errors="coerce")
        elif pre_Y:
            PRE["_LP"] = np.exp(pd.to_numeric(PRE[pre_Y], errors="coerce"))
        else:
            raise ValueError("pre-panel has neither LP nor Y=ln(LP).")

        C = PRE[[pre_port,"_qtuple","_LP"]].dropna().rename(columns={pre_port:port})
        C["_source"] = "pre_panel_fallback"
    else:
        C = pd.DataFrame(columns=[port,"_qtuple","_LP","_source"])

    # Combine with priority A > B > C
    combined = pd.concat([A, B, C], ignore_index=True)
    if combined.empty:
        raise RuntimeError("No port×quarter LP rows from any source.")

    key = [port,"_qtuple"]
    pref = {"LP_panel_port_quarter": 2, "LP_panel_termQ_to_portQ": 1, "pre_panel_fallback": 0}
    combined["_pref"] = combined["_source"].map(pref).fillna(-1)
    combined = (combined.sort_values(key+["_pref"], ascending=[True, True, False])
                        .drop_duplicates(subset=key, keep="first")
                        .drop(columns=["_pref"]))

    # Decorate & write
    out = combined.copy()
    out["qtr"] = out["_qtuple"].apply(lambda t: f"{int(t[0])}Q{int(t[1])}")
    out = out.rename(columns={port: "port", "_LP": "LP"})[["port","qtr","LP","_source"]]
    out = out.sort_values(["port","qtr"]).reset_index(drop=True)

    out.to_csv(out_csv, index=False)
    meta = {
        "inputs": {"LP_Panel": str(p_lp), "pre_panel": str(p_pre)},
        "outputs": {"csv": str(out_csv)},
        "counts": {
            "LP_panel_rows": int(len(LP)),
            "A_portQ_rows": int(len(A)),
            "B_termQ_to_portQ_rows": int(len(B)),
            "C_pre_rows": int(len(C)),
            "final_rows": int(len(out)),
        },
        "sources_used": sorted(out["_source"].unique().tolist()),
        "ports": sorted(out["port"].unique().tolist()),
        "qtr_min": out["qtr"].min() if len(out) else None,
        "qtr_max": out["qtr"].max() if len(out) else None,
    }
    Path(out_meta).write_text(json.dumps(meta, indent=2))
    print(f"[combine] WROTE: {out_csv}")
    print(f"[combine] WROTE: {out_meta}")

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    main()
