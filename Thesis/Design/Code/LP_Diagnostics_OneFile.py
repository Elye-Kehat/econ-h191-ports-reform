#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LP_Diagnostics_OneFile.py
-------------------------
Read LP_Panel.tsv and write ONE text report that contains all diagnostics,
with TSV tables under markdown-style section headers.

Example:
  python LP_Diagnostics_OneFile.py \
    --panel "Data/LP/LP_Panel.tsv" \
    --out   "Design/Output Data/diagnostics/lp_diagnostics_report.txt"

What’s inside the report (all TSV):
  1) Series Inventory
  2) Integrity LP_vs_wPi (max rel err)
  3) mean(w) by (freq,port,year), sorted by |w-1|
  4) Port-Quarter w span across terminals (+ rows where w differs)
  5) Pairwise contrasts: entrant vs. legacy (Haifa, Ashdod)
  6) Pi audit by terminal-year
  7) LP mean by terminal (all vs. excl. 2023Q4)
  8) Pre-reform LP vs LP_id (corr & MAPE) for monthly ports
  9) Sanity counts (monthly rows after 2021-08; zero LP rows)
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

QMAP = {"Q1":3, "Q2":6, "Q3":9, "Q4":12}

def to_datetime(row):
    if row["freq"] == "M":
        y = int(row["year"])
        m = int(row["month"]) if pd.notna(row["month"]) else 1
        return pd.Timestamp(year=y, month=m, day=1)
    q = str(row["quarter"]).strip()
    if q not in QMAP:
        return pd.NaT
    return pd.Timestamp(year=int(row["year"]), month=QMAP[q], day=1)

def df_to_tsv(df: pd.DataFrame) -> str:
    return df.to_csv(sep="\t", index=False).strip()

def write_section(fh, title: str, payload):
    fh.write(f"### {title}\n")
    if isinstance(payload, pd.DataFrame):
        fh.write(df_to_tsv(payload) + "\n\n")
    else:
        fh.write(str(payload).rstrip() + "\n\n")

def load_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    needed = {"series_id","freq","port","year","LP","w","Pi"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"[LP_Panel] missing required columns: {sorted(missing)}")
    df["date"] = df.apply(to_datetime, axis=1)
    return df

def pairwise_contrast(dfq, port, entrant, legacy):
    a = dfq[(dfq["port"]==port) & (dfq["terminal"]==entrant)][
        ["year","quarter","LP","Pi","w"]
    ].rename(columns={"LP":f"LP_{entrant}","Pi":f"Pi_{entrant}","w":f"w_{entrant}"})
    b = dfq[(dfq["port"]==port) & (dfq["terminal"]==legacy)][
        ["year","quarter","LP","Pi","w"]
    ].rename(columns={"LP":f"LP_{legacy}","Pi":f"Pi_{legacy}","w":f"w_{legacy}"})
    m = pd.merge(a,b,on=["year","quarter"],how="inner")
    if m.empty:
        return m
    m["LP_diff"] = m[f"LP_{entrant}"] - m[f"LP_{legacy}"]
    m["Pi_diff"] = m[f"Pi_{entrant}"] - m[f"Pi_{legacy}"]
    m["w_diff"]  = m[f"w_{entrant}"]  - m[f"w_{legacy}"]
    return m.sort_values(["year","quarter"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="Path to Data/LP/LP_Panel.tsv")
    ap.add_argument("--out",   required=True, help="Path to write the single TXT report")
    ap.add_argument("--war_q", default="2023Q4",
                    help='Quarter to exclude in "no-war" averages (format YYYYQ#). Default: 2023Q4')
    # terminal labels — override if your canonical names differ
    ap.add_argument("--haifa_entrant", default="Haifa-Bayport")
    ap.add_argument("--haifa_legacy",  default="Haifa-Legacy")
    ap.add_argument("--ashdod_entrant", default="Ashdod-HCT")
    ap.add_argument("--ashdod_legacy",  default="Ashdod-Legacy")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df = load_panel(args.panel)

    with open(args.out, "w") as fh:
        write_section(fh, "Report",
            "LP Diagnostics (single-file). All tables below are TSV, copy/paste ready.")

        # 1) Series inventory
        series = (df.groupby(["series_id","freq","port"], dropna=False)
                    .agg(n_rows=("LP","size"),
                         date_min=("date","min"),
                         date_max=("date","max"))
                    .reset_index()
                    .sort_values(["port","series_id"]))
        write_section(fh, "Series Inventory", series)

        # 2) Integrity: LP ≈ w × Π
        prod = df["w"] * df["Pi"]
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.where(df["LP"]!=0,
                           np.abs(df["LP"]-prod)/np.maximum(np.abs(df["LP"]),1e-12),
                           0.0)
        write_section(fh, "Integrity LP_vs_wPi",
                      f"max_relative_error\t{np.nanmax(rel):.3e}")

        # 3) mean(w) by (freq,port,year)
        w_means = (df.groupby(["freq","port","year"], dropna=False)
                     .agg(w_mean=("w","mean"))
                     .reset_index())
        w_means["abs_dev"] = (w_means["w_mean"] - 1.0).abs()
        w_means = w_means.sort_values(["abs_dev","freq","port","year"],
                                      ascending=[False,True,True,True])
        write_section(fh, "QA mean(w) by (freq,port,year) — sorted by |w_mean-1|", w_means)

        # 4) Port-quarter: is w shared across terminals?
        q = df[df["freq"]=="Q"].copy()
        shared = pd.DataFrame()
        if "terminal" in q.columns and not q.empty:
            shared = (q.groupby(["port","year","quarter"], dropna=False)
                        .agg(w_min=("w","min"), w_max=("w","max"),
                             w_std=("w","std"), n=("w","size"))
                        .reset_index())
            shared["w_span"] = shared["w_max"] - shared["w_min"]
            shared["shared_flag"] = np.where(shared["w_span"].abs()<=1e-9, "shared", "differs")
            write_section(fh, "Port-Quarter w span across terminals",
                          shared.sort_values(["port","year","quarter"]))
            write_section(fh, "Rows where w differs across terminals",
                          shared.query("shared_flag=='differs'").sort_values("w_span", ascending=False))

        # 5) Pairwise contrasts entrant vs legacy
        def safe_contrast(port, entrant, legacy):
            if q.empty or "terminal" not in q.columns:
                return pd.DataFrame()
            present = set(q[q["port"]==port]["terminal"].unique())
            if entrant not in present or legacy not in present:
                return pd.DataFrame()
            return pairwise_contrast(q, port, entrant, legacy)

        haifa_tab  = safe_contrast("Haifa",  args.haifa_entrant,  args.haifa_legacy)
        ashdod_tab = safe_contrast("Ashdod", args.ashdod_entrant, args.ashdod_legacy)
        write_section(fh, f"Contrast — {args.haifa_entrant} vs {args.haifa_legacy} (quarterly)", haifa_tab)
        write_section(fh, f"Contrast — {args.ashdod_entrant} vs {args.ashdod_legacy} (quarterly)", ashdod_tab)

        # 6) Pi audit by terminal-year
        if not q.empty:
            pi_audit = (q.groupby(["port","terminal","year"], dropna=False)["Pi"]
                          .agg(Pi_mean=("mean"), Pi_median=("median"),
                               Pi_min=("min"), Pi_max=("max"), nQ=("size"))
                          .reset_index()
                          .sort_values(["port","terminal","year"]))
            write_section(fh, "Pi audit by terminal-year", pi_audit)

        # 7) LP means by terminal (all vs exclude war quarter)
        war_year, war_quarter = 2023, "Q4"
        try:
            war_year = int(args.war_q[:4])
            war_quarter = args.war_q[4:]
        except Exception:
            pass
        if not q.empty:
            q["war_q"] = ((q["year"]==war_year) & (q["quarter"].astype(str)==war_quarter)).astype(int)
            lp_all = (q.groupby(["port","terminal"])
                        .agg(LP_mean=("LP","mean"), n=("LP","size")).reset_index())
            lp_no_war = (q[q["war_q"]==0].groupby(["port","terminal"])
                           .agg(LP_mean=("LP","mean"), n=("LP","size")).reset_index())
            write_section(fh, "LP mean by terminal — all quarters", lp_all)
            write_section(fh, f"LP mean by terminal — exclude {war_year}{war_quarter}", lp_no_war)

        # 8) Pre-reform identity check (monthly only)
        m = df[(df["freq"]=="M") & (df["date"]<=pd.Timestamp("2021-08-31"))].copy()
        if "LP_id" in m.columns and not m.empty:
            def corr_mape(g):
                sub = g.dropna(subset=["LP","LP_id"]).copy()
                if len(sub) < 3:
                    return pd.Series({"corr": np.nan, "mape": np.nan, "n": len(sub)})
                c = sub["LP"].corr(sub["LP_id"])
                mape = (np.abs(sub["LP"] - sub["LP_id"]) /
                        np.maximum(np.abs(sub["LP_id"]), 1e-12)).mean()
                return pd.Series({"corr": c, "mape": mape, "n": len(sub)})
            id_tab = m.groupby(["port"], dropna=False).apply(corr_mape).reset_index()
            write_section(fh, "Pre-reform LP vs LP_id (port-monthly)", id_tab)

        # 9) Sanity counts
        post_m_ct = len(df[(df["freq"]=="M") & (df["date"]>pd.Timestamp("2021-08-31"))])
        zero_lp   = int(np.isclose(df["LP"].fillna(0), 0.0).sum())
        sanity = pd.DataFrame({"check": ["monthly_rows_after_2021_08","zero_LP_count"],
                               "value": [post_m_ct, zero_lp]})
        write_section(fh, "Sanity counts", sanity)

    print(f"[ok] wrote: {args.out}")

if __name__ == "__main__":
    main()
