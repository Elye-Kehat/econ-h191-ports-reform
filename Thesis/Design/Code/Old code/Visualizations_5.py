#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualizations v5 — resilient & data-driven
-------------------------------------------
- Input: LP_panel_mixedfreq.tsv (mixed-frequency panel built by v3a)
- Auto-finds input but allows --panel override
- Skips Eilat
- For each port (Haifa, Ashdod):
  * Overview: TEU, tons, L_hours, LP_port (LP_mix at port level), each indexed to 100
  * LP-only split: pre-reform port LP; post-reform terminal LPs (auto-detected terminals);
    each segment indexed to 100 using its own first non-null
- Hard y-limit [0, 300]
- Writes a 'visual_audit.tsv' with counts/availability to help debug missing lines
"""

import argparse, os
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- file discovery --------------------------------

DEFAULT_CANDIDATES = [
    "Design/Code/Input Data/LP_panel_mixedfreq copy.tsv",
    "Design/Code/Input Data/LP_panel_mixedfreq.tsv",
    "Data/LP/LP_panel_mixedfreq.tsv",
    "LP_panel_mixedfreq.tsv",
]

def find_first_file(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

# ----------------------------- helpers ---------------------------------------

def norm_port(s: str) -> str:
    if s is None:
        return s
    lo = str(s).replace("–","-").lower()
    if "haifa" in lo: return "Haifa"
    if "ashdod" in lo: return "Ashdod"
    if "eilat" in lo: return "Eilat"
    return str(s).strip()

def ym_to_ts(y, m):
    try:
        return pd.Timestamp(int(y), int(m), 1)
    except Exception:
        return pd.NaT

def index_100(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s
    base = s.dropna().iloc[0]
    if pd.isna(base) or base == 0:
        return s * np.nan
    return (s / base) * 100.0

def read_panel(path: Optional[str]) -> pd.DataFrame:
    p = path or find_first_file(DEFAULT_CANDIDATES)
    if not p:
        raise FileNotFoundError("Could not find LP_panel_mixedfreq.tsv. Pass --panel to specify path.")
    df = pd.read_csv(p, sep="\t", engine="python")
    # tolerant header mapping
    def pick(cols, wanted):
        for c in cols:
            if c.lower() == wanted.lower():
                return c
        for c in cols:
            if wanted.lower() in c.lower():
                return c
        return None

    colmap = {}
    for w in ["level","port","terminal","year","month","quarter","freq","TEU","tons","w","w_source","Pi","L_hours","LP_mix","LP_id"]:
        k = pick(df.columns, w) or w
        if k in df.columns:
            colmap[w] = k
    g = df[[colmap[c] for c in colmap]].rename(columns={v:k for k,v in colmap.items()})
    # normalize
    g["port"] = g["port"].astype(str).map(norm_port)
    g["terminal"] = g.get("terminal", pd.Series([None]*len(g))).astype(str).replace({"nan":None})
    g["year"] = pd.to_numeric(g["year"], errors="coerce").astype("Int64")
    g["month"] = pd.to_numeric(g["month"], errors="coerce").astype("Int64")
    g["date"] = [ym_to_ts(y, m) for y,m in zip(g["year"], g["month"])]
    # numeric
    for c in ["TEU","tons","w","Pi","L_hours","LP_mix","LP_id"]:
        if c not in g.columns: g[c] = np.nan
        g[c] = pd.to_numeric(g[c], errors="coerce")
    # drop bad dates
    g = g[g["date"].notna()].copy()
    # collapse duplicates
    agg = {c:"mean" for c in ["TEU","tons","w","Pi","L_hours","LP_mix","LP_id"]}
    g = g.groupby(["level","port","terminal","year","month","date","freq","quarter"], dropna=False, as_index=False).agg(agg)
    return g

def earliest_terminal_date(df_term: pd.DataFrame) -> Optional[pd.Timestamp]:
    dt = df_term[df_term["LP_mix"].notna()].sort_values("date")["date"]
    return None if dt.empty else dt.iloc[0]

def plot_overview(port_df: pd.DataFrame, port: str, outdir: str):
    d = port_df.copy().sort_values("date")
    series = {
        "TEU (index=100)": index_100(d["TEU"]),
        "tons (index=100)": index_100(d["tons"]),
        "L_hours (index=100)": index_100(d["L_hours"]),
        "LP_port (index=100)": index_100(d["LP_mix"]),
    }
    plt.figure(figsize=(14,4))
    plotted = False
    colors = [None, None, "#2ca02c", "#d62728"]
    for (lab, s), c in zip(series.items(), colors):
        if s.notna().any():
            plt.plot(d["date"], s, label=lab, color=c)
            plotted = True
    plt.ylim(0,300)
    plt.title(f"{port} — TEU, tons, L, LP (indexed to 100 at first available)")
    plt.xlabel("Date"); plt.ylabel("Index (100=first observed)")
    if plotted:
        plt.legend()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fp = os.path.join(outdir, f"Overview_{port}.png")
    plt.savefig(fp, bbox_inches="tight", dpi=160)
    plt.close()
    print(f"[wrote] {fp}")

def plot_lp_split(port_rows: pd.DataFrame, term_rows: pd.DataFrame, port: str, outdir: str, cutover_hint: Optional[str]):
    # determine cutover: earliest terminal LP date or hint
    cut = earliest_terminal_date(term_rows)
    if cut is None and cutover_hint:
        try:
            y, m = cutover_hint.split("-")
            cut = pd.Timestamp(int(y), int(m), 1)
        except Exception:
            cut = None
    if cut is None:
        # nothing to split
        print(f"[warn] No terminal LP for {port}; skipping LP-only split.")
        return

    pre = port_rows[(port_rows["date"] < cut) & port_rows["LP_mix"].notna()].sort_values("date")
    post = term_rows[(term_rows["date"] >= cut) & term_rows["LP_mix"].notna()].copy()

    # pick 1-2 most available terminals in post
    counts = (post.groupby("terminal")["LP_mix"].apply(lambda x: x.notna().sum())
              .sort_values(ascending=False))
    keep_terms = counts.index.tolist()[:2]

    plt.figure(figsize=(14,4))
    plotted = False
    if not pre.empty:
        plt.plot(pre["date"], index_100(pre["LP_mix"]), label="LP_port (pre-reform, index=100)", color="#9467bd")
        plotted = True
    for t in keep_terms:
        dt = post[post["terminal"]==t].sort_values("date")
        if not dt.empty:
            plt.plot(dt["date"], index_100(dt["LP_mix"]), label=f"LP_{t.split('-')[-1]} (index=100)")
            plotted = True
    plt.ylim(0,300)
    plt.title(f"{port} — LP series (pre-reform port + post-reform terminals)")
    plt.xlabel("Date"); plt.ylabel("Index (100=first observed)")
    if plotted:
        plt.legend()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fp = os.path.join(outdir, f"LP_only_split_{port}.png")
    plt.savefig(fp, bbox_inches="tight", dpi=160)
    plt.close()
    print(f"[wrote] {fp}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=None, help="Path to LP_panel_mixedfreq.tsv")
    ap.add_argument("--out_dir", default="Design/Code/Output Data/visuals_5", help="Where to write PNGs and audit TSV")
    ap.add_argument("--ports", default="Haifa,Ashdod", help="Comma list; Eilat ignored even if passed")
    ap.add_argument("--cutover", default="Haifa:2021-09,Ashdod:2022-07", help="Optional fallback cutover YYYY-MM")
    args = ap.parse_args()

    panel = read_panel(args.panel)

    # restrict ports
    want = [p.strip() for p in args.ports.split(",") if p.strip()]
    want = [p for p in want if p in {"Haifa","Ashdod"}]  # forcibly drop Eilat
    panel = panel[panel["port"].isin(want)].copy()

    # split level dfs
    port_df = panel[panel["level"]=="port"].copy()
    term_df = panel[panel["level"]=="terminal"].copy()

    # audit table
    rows = []
    for p in want:
        pr = port_df[port_df["port"]==p]
        tr = term_df[term_df["port"]==p]
        rows.append({
            "port": p,
            "port_rows": len(pr),
            "port_LP_mix_nonnull": int(pr["LP_mix"].notna().sum()),
            "term_rows": len(tr),
            "term_LP_mix_nonnull": int(tr["LP_mix"].notna().sum()),
            "terminals_seen": ", ".join(sorted(set(tr["terminal"].dropna().unique().tolist()))[:5])
        })
    audit = pd.DataFrame(rows)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    audit_fp = os.path.join(args.out_dir, "visual_audit.tsv")
    audit.to_csv(audit_fp, sep="\t", index=False)
    print(f"[wrote] {audit_fp}")

    # cutover map from hint
    hint = {}
    for kv in str(args.cutover).split(","):
        if ":" in kv:
            k,v = kv.split(":",1)
            hint[k.strip()] = v.strip()

    # plots
    for p in want:
        pr = port_df[port_df["port"]==p]
        tr = term_df[term_df["port"]==p]
        plot_overview(pr, p, args.out_dir)
        plot_lp_split(pr, tr, p, args.out_dir, hint.get(p))

if __name__ == "__main__":
    main()
