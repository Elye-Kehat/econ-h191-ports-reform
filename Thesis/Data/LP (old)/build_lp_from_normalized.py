#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_lp_from_normalized.py
Reads canonical normalized inputs from --norm_dir and builds LP outputs.
No header guessing; schema is fixed by the normalizer.

Outputs (in --out_dir):
- LP_port_month_mixadjusted.tsv
- LP_port_month_identity.tsv
- LP_terminal_month_mixadjusted.tsv
- LP_terminal_quarter_mixadjusted.tsv
- LP_panel_mixedfreq.tsv
- qa_lp_report.tsv
- _meta_lp_mixadjusted.json
"""

import argparse, os, json
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", engine="python")

def winsorize_group(df: pd.DataFrame, value_col: str, by: List[str], lower=0.01, upper=0.99) -> pd.Series:
    if df.empty: return df[value_col].copy()
    out = df[value_col].astype(float).copy()
    g = df.groupby(by, dropna=False, sort=False)
    qs = g[value_col].quantile([lower, upper]).unstack(level=-1)
    qs = qs.rename(columns={lower:"q_low", upper:"q_high"})
    key = pd.MultiIndex.from_frame(df[by])
    ql = qs.reindex(key)["q_low"].to_numpy()
    qh = qs.reindex(key)["q_high"].to_numpy()
    v = out.to_numpy(dtype="float64")
    v = np.where(~np.isnan(v) & ~np.isnan(ql), np.maximum(v, ql), v)
    v = np.where(~np.isnan(v) & ~np.isnan(qh), np.minimum(v, qh), v)
    return pd.Series(v, index=df.index, dtype="float64")

def _quarter_from_month(m) -> Optional[str]:
    if pd.isna(m): return None
    try:
        q = (int(m)-1)//3 + 1
        return f"Q{q}"
    except Exception:
        return None

def compute_w(tons_pm: pd.DataFrame, teu_pm: pd.DataFrame, teu_pq: pd.DataFrame,
              winsor_lower: float, winsor_upper: float) -> pd.DataFrame:
    # monthly path
    w_m = tons_pm.merge(teu_pm, on=["port","year","month","month_index"], how="left")
    w_m["tons_per_teu"] = np.where(w_m["teu_p_m"]>0, w_m["tons_p_m"]/w_m["teu_p_m"], np.nan)
    w_m["r_win"] = winsorize_group(w_m, "tons_per_teu", ["port","year"], winsor_lower, winsor_upper)
    m_mean = w_m.groupby(["port","year"], dropna=False)["r_win"].transform("mean")
    w_m["w_p_m"] = np.where((m_mean==0)|m_mean.isna(), 1.0, w_m["r_win"]/m_mean)
    w_m["w_src_monthly"] = np.where(w_m["tons_per_teu"].notna(), "monthly", None)

    # quarterly fallback
    tons_q = tons_pm.copy()
    tons_q["quarter"] = tons_q["month"].apply(_quarter_from_month).astype(object)
    tq = tons_q.groupby(["port","year","quarter"], dropna=False)["tons_p_m"].sum(min_count=1).reset_index()
    rq = tq.merge(teu_pq, on=["port","year","quarter"], how="left")
    rq["r_q"] = np.where(rq["teu_p_q"]>0, rq["tons_p_m"]/rq["teu_p_q"], np.nan)
    rq["r_q_win"] = winsorize_group(rq, "r_q", ["port","year"], winsor_lower, winsor_upper)
    q_mean = rq.groupby(["port","year"], dropna=False)["r_q_win"].transform("mean")
    rq["w_p_q"] = np.where((q_mean==0)|q_mean.isna(), 1.0, rq["r_q_win"]/q_mean)

    # broadcast to months present in tons_pm
    map_m = tons_pm[["port","year","month","month_index"]].drop_duplicates().copy()
    map_m["quarter"] = map_m["month"].apply(_quarter_from_month).astype(object)
    w_qm = map_m.merge(rq[["port","year","quarter","w_p_q"]], on=["port","year","quarter"], how="left")
    w_qm = w_qm.rename(columns={"w_p_q":"w_from_q"})
    w_qm["w_src_quarterly"] = np.where(w_qm["w_from_q"].notna(), "quarterly", None)

    # final
    wf = w_m.merge(w_qm, on=["port","year","month","month_index"], how="outer", validate="one_to_one")
    wf["w_final"] = wf["w_p_m"].combine_first(wf["w_from_q"])
    wf["w_source"] = wf["w_src_monthly"].astype(object).combine_first(wf["w_src_quarterly"].astype(object))
    wf["w_source"] = wf["w_source"].astype(object)
    return wf[["port","year","month","month_index","w_final","w_source"]]

def build_pi_mix(l_proxy: pd.DataFrame, months_key: pd.DataFrame) -> pd.DataFrame:
    lp = l_proxy.copy()
    lp["quarter"] = lp["month"].apply(_quarter_from_month).astype(object)
    teui = (lp.groupby(["port","terminal","year","quarter"], dropna=False)["teu_i_m"]
              .sum(min_count=1).reset_index().rename(columns={"teu_i_m":"teu_i_q_sum"}))
    teutot = (teui.groupby(["port","year","quarter"], dropna=False)["teu_i_q_sum"]
                .sum(min_count=1).reset_index().rename(columns={"teu_i_q_sum":"teu_port_q"}))
    shares = teui.merge(teutot, on=["port","year","quarter"], how="left")
    shares["share_i_q"] = np.where(shares["teu_port_q"]>0, shares["teu_i_q_sum"]/shares["teu_port_q"], np.nan)
    pi_i_y = (lp.groupby(["port","terminal","year"], dropna=False)["pi_teu_per_hour_i_y"].first().reset_index())
    shares = shares.merge(pi_i_y, on=["port","terminal","year"], how="left")
    pi_q = (shares.assign(pi_w=lambda d: d["share_i_q"]*d["pi_teu_per_hour_i_y"])
                 .groupby(["port","year","quarter"], dropna=False)["pi_w"].sum(min_count=1).reset_index()
                 .rename(columns={"pi_w":"Pi_p_q"}))
    months_key = months_key.copy()
    months_key["quarter"] = months_key["month"].apply(_quarter_from_month).astype(object)
    pi_pm = months_key.merge(pi_q, on=["port","year","quarter"], how="left")
    return pi_pm.rename(columns={"Pi_p_q":"pi_p_y_mixbase"})

def build_outputs(norm_dir: str, out_dir: str, cutover_map: Dict[str,str], winsor_lower: float, winsor_upper: float):
    tons_pm = _read(os.path.join(norm_dir, "tons_port_month.tsv"))
    teu_pm  = _read(os.path.join(norm_dir, "teu_port_month.tsv"))
    teu_pq  = _read(os.path.join(norm_dir, "teu_port_quarter.tsv"))
    lproxy  = _read(os.path.join(norm_dir, "l_proxy.tsv"))

    # compute w
    w = compute_w(tons_pm, teu_pm, teu_pq, winsor_lower, winsor_upper)

    # port LP
    pi_pm = build_pi_mix(lproxy, w[["port","year","month","month_index"]].drop_duplicates())
    lp_port = (w.merge(pi_pm, on=["port","year","month","month_index"], how="left")
                 .merge(tons_pm, on=["port","year","month","month_index"], how="left"))
    lp_port["lp_port_month_mix"] = lp_port["w_final"] * lp_port["pi_p_y_mixbase"]

    # identity LP (sparse)
    L_port_m = (lproxy.groupby(["port","year","month"], dropna=False)["l_hours_i_m"].sum(min_count=1)
                      .reset_index().rename(columns={"l_hours_i_m":"l_port_m"}))
    lp_id = L_port_m.merge(teu_pm, on=["port","year","month"], how="left")
    lp_id["lp_port_month_id"] = np.where(lp_id["l_port_m"]>0, lp_id["teu_p_m"]/lp_id["l_port_m"], np.nan)

    lp_port = lp_port.merge(L_port_m, on=["port","year","month"], how="left")
    lp_port = lp_port[["port","year","month","month_index","teu_p_m","tons_p_m","w_final","w_source",
                       "pi_p_y_mixbase","lp_port_month_mix","l_port_m","tons_source"]].copy()

    # terminal monthly LP
    term_m = lproxy.merge(w[["port","year","month","w_final"]], on=["port","year","month"], how="left")
    term_m["lp_term_month_mixadjusted"] = term_m["pi_teu_per_hour_i_y"] * term_m["w_final"]
    bad = (pd.to_numeric(term_m["teu_i_m"], errors="coerce")<=0) | (pd.to_numeric(term_m["l_hours_i_m"], errors="coerce")<=0)
    term_m.loc[bad, "lp_term_month_mixadjusted"] = np.nan
    term_m = term_m[["port","terminal","year","month","month_index","quarter","pi_teu_per_hour_i_y",
                     "w_final","teu_i_m","l_hours_i_m","lp_term_month_mixadjusted"]].copy()

    # terminal quarterly mixedfreq after cutover
    cut_idx = {}
    for p, ym in cutover_map.items():
        try:
            y, m = ym.split("-")
            cut_idx[p] = int(y)*12 + int(m)
        except Exception:
            cut_idx[p] = 10**9
    t = term_m.copy()
    t["freq"] = np.where(t["port"].map(cut_idx).le(t["month_index"]), "Q", "M")
    term_M = t[t["freq"]=="M"].copy()
    term_Q = t[t["freq"]=="Q"].copy()
    if not term_Q.empty:
        qagg = term_Q.groupby(["port","terminal","year","quarter"], dropna=False).agg(
            pi_teu_per_hour_i_y=("pi_teu_per_hour_i_y","first"),
            w_final=("w_final","mean"),
            teu_i_m=("teu_i_m","sum"),
            l_hours_i_m=("l_hours_i_m","sum"),
            lp_term_month_mixadjusted=("lp_term_month_mixadjusted","mean")
        ).reset_index()
        q_to_m = {"Q1":3,"Q2":6,"Q3":9,"Q4":12}
        qagg["month"] = qagg["quarter"].map(q_to_m).astype("Int64")
        qagg["month_index"] = (qagg["year"].astype(int)*12 + qagg["month"].astype(int)).astype(int)
        qagg["freq"] = "Q"
        term_qview = qagg[["port","terminal","year","quarter","month","month_index","freq",
                           "pi_teu_per_hour_i_y","w_final","teu_i_m","l_hours_i_m","lp_term_month_mixadjusted"]]
    else:
        term_qview = pd.DataFrame(columns=["port","terminal","year","quarter","month","month_index","freq",
                           "pi_teu_per_hour_i_y","w_final","teu_i_m","l_hours_i_m","lp_term_month_mixadjusted"])

    # panel
    port_panel = lp_port.copy()
    port_panel["level"] = "port"; port_panel["terminal"] = pd.NA
    port_panel["Pi"] = port_panel["pi_p_y_mixbase"]; port_panel["L_hours"] = port_panel["l_port_m"]
    port_panel["LP_mix"] = port_panel["lp_port_month_mix"]; port_panel["LP_id"] = pd.NA
    port_panel["TEU"] = port_panel["teu_p_m"]; port_panel["tons"] = port_panel["tons_p_m"]
    port_panel["w"] = port_panel["w_final"]; port_panel["freq"] = "M"
    port_panel["quarter"] = port_panel["month"].apply(_quarter_from_month)
    port_panel = port_panel[["level","port","terminal","year","month","month_index","quarter","freq",
                             "TEU","tons","w","w_source","Pi","L_hours","LP_mix","LP_id","tons_source"]]

    term_panel = term_qview.rename(columns={
        "pi_teu_per_hour_i_y":"Pi","l_hours_i_m":"L_hours","lp_term_month_mixadjusted":"LP_mix",
        "teu_i_m":"TEU","w_final":"w"
    }).copy()
    term_panel["level"]="terminal"; term_panel["LP_id"]=pd.NA; term_panel["tons"]=pd.NA; term_panel["w_source"]=pd.NA; term_panel["tons_source"]=pd.NA
    term_panel = term_panel[["level","port","terminal","year","month","month_index","quarter","freq",
                             "TEU","tons","w","w_source","Pi","L_hours","LP_mix","LP_id","tons_source"]]

    panel = pd.concat([port_panel, term_panel], ignore_index=True).sort_values(["level","port","terminal","year","month"]).reset_index(drop=True)

    # QA
    qa = []
    def _uniq(df, keys, name):
        dup = df.duplicated(keys).sum()
        qa.append({"check":f"unique_{name}", "result":"pass" if dup==0 else "fail", "detail":f"dups={dup} keys={keys}"})
    _uniq(lp_port, ["port","year","month"], "lp_port")
    _uniq(term_m, ["port","terminal","year","month"], "term_m")
    _uniq(w, ["port","year","month"], "w")

    ann = lp_port.groupby(["port","year"], dropna=False).agg(mu_lp=("lp_port_month_mix","mean"),
                                                             mu_pi=("pi_p_y_mixbase","mean")).reset_index()
    ann["rel_err"] = np.abs(ann["mu_lp"]-ann["mu_pi"])/ann["mu_pi"].replace(0, np.nan)
    for _,r in ann.iterrows():
        qa.append({"check":"annual_preservation","port":r["port"],"year":int(r["year"]),
                   "mu_lp":float(r["mu_lp"]) if pd.notna(r["mu_lp"]) else None,
                   "mu_pi":float(r["mu_pi"]) if pd.notna(r["mu_pi"]) else None,
                   "rel_err":float(r["rel_err"]) if pd.notna(r["rel_err"]) else None,
                   "result":"pass" if (pd.isna(r["rel_err"]) or r["rel_err"]<=1e-6) else "warn"})

    qa_df = pd.DataFrame(qa)

    # write
    os.makedirs(out_dir, exist_ok=True)
    def _w(df, name):
        p = os.path.join(out_dir, name); df.to_csv(p, sep="\t", index=False); return p

    p1 = _w(lp_port, "LP_port_month_mixadjusted.tsv")
    p2 = _w(lp_id, "LP_port_month_identity.tsv")
    p3 = _w(term_m, "LP_terminal_month_mixadjusted.tsv")
    p4 = _w(term_qview, "LP_terminal_quarter_mixadjusted.tsv")
    p5 = _w(panel, "LP_panel_mixedfreq.tsv")
    p6 = _w(qa_df, "qa_lp_report.tsv")

    meta = {"winsor":{"lower":winsor_lower,"upper":winsor_upper},
            "cutover":cutover_map, "norm_dir":norm_dir,
            "rows":{"lp_port":len(lp_port),"lp_id":len(lp_id),"term_m":len(term_m),"term_q":len(term_qview),"panel":len(panel)}}
    with open(os.path.join(out_dir, "_meta_lp_mixadjusted.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[build] Wrote outputs to", out_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm_dir", required=True, help="Directory with normalized TSVs from the normalizer")
    ap.add_argument("--out_dir", default="Data/LP", help="Output directory")
    ap.add_argument("--cutover", default="Haifa:2021-09,Ashdod:2022-07", help="Comma list like 'Haifa:2021-09,Ashdod:2022-07'")
    ap.add_argument("--winsor_lower", type=float, default=0.01)
    ap.add_argument("--winsor_upper", type=float, default=0.99)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    # parse cutover
    cut = {}
    if args.cutover:
        for kv in str(args.cutover).split(","):
            if ":" in kv:
                k,v = kv.split(":",1); cut[k.strip()] = v.strip()
    if args.validate_only:
        # Just check files exist
        req = ["tons_port_month.tsv","teu_port_month.tsv","teu_port_quarter.tsv","l_proxy.tsv"]
        missing = [r for r in req if not os.path.exists(os.path.join(args.norm_dir, r))]
        if missing:
            raise SystemExit(f"[validate-only] Missing normalized files: {missing}")
        print("[validate-only] Normalized inputs present.")
        raise SystemExit(0)

    build_outputs(args.norm_dir, args.out_dir, cut, args.winsor_lower, args.winsor_upper)

if __name__ == "__main__":
    main()
