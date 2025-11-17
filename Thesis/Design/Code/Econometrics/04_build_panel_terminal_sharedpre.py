#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build terminal x quarter (logs) panel with SHARED PRE per terminal.
- Reads the enriched step-3 TSV (has qtr, Y=ln(lp), t_index).
- Detects PRE rows (treat NaN/aliases as PRE), duplicates them to Legacy + Entrant per port.
- Normalizes compound terminal labels to canonical {Legacy, SIPG, HCT} and keeps observed POST rows.
- Stacks POST first THEN PRE (so observed wins on overlaps), sorts, de-dups.
- Optionally attaches event clocks (tau_comp, tau_priv) from model1_params.yaml.
- Computes nyt_ok_comp if both competition dates exist.
- Writes final CSV + robust meta JSON.
- Optional: --debug emits two compact debug artifacts.
"""
import argparse, json, sys, re
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
try:
    import yaml
except Exception:
    yaml = None
STEP3_CANDIDATES = [
    Path("Design/Output Data/03_LP_Panel_quarterized.step3_qtr_Y_tindex.tsv"),
    Path("Design/Output Data/LP_Panel_quarterized.step3_qtr_Y_tindex.tsv"),
    Path("/mnt/data/LP_Panel_quarterized.step3_qtr_Y_tindex.tsv"),
    Path("/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis/Design/Output Data/03_LP_Panel_quarterized.step3_qtr_Y_tindex.tsv")
]

YAML_CANDIDATES = [
    Path("Design/Code/Econometrics/00_model1_params.yaml"),
    Path("Design/Code/Econometrics/model1_params.yaml"),
    Path("/mnt/data/model1_params.yaml"),
]

KEEP_TERMS_CANON = {"Legacy","SIPG","HCT"}
PORT_ENTRANT = {"Haifa":"SIPG","Ashdod":"HCT"}
PORT_TITLE = lambda s: str(s).strip().title() if pd.notna(s) else s
def find_input(cands):
    for p in cands:
        if p.exists():
            return p, p.parent
    sys.exit("Could not find step-3 TSV in expected locations. Aborting.")
def load_yaml_optional():
    if yaml is None:
        return None
    for y in YAML_CANDIDATES:
        if y.exists():
            try:
                return yaml.safe_load(open(y,"r",encoding="utf-8"))
            except Exception:
                return None
    return None
def norm_terminal(val):
    if val is None or (isinstance(val,float) and np.isnan(val)):
        return ""
    raw = str(val).strip()
    if raw == "":
        return ""
    u = re.sub(r"[-_]", " ", raw.upper())
    u = re.sub(r"\s+", " ", u).strip()
    if u in {"PORT","SUM"}:
        return ""
    if any(k in u for k in ["SUM TERMINALS","SUMTERMINALS","PORT LEVEL","PORTLEVEL","TOTAL","AGGREGATE"]):
        return ""
    if ("LEGACY" in u) or ("HPC" in u) or ("APC" in u):
        return "Legacy"
    if any(k in u for k in ["SIPG","BAYPORT","BAY PORT","HAIFA BAYPORT"]):
        return "SIPG"
    if any(k in u for k in ["HCT","SOUTHPORT","SOUTH PORT","TIL","ASHDOD SOUTHPORT"]):
        return "HCT"
    last = u.split()[-1]
    if last in {"LEGACY","SIPG","HCT"}:
        return "Legacy" if last=="LEGACY" else last
    return raw
def expected_entrant_for_port(port, terminals_seen):
    p = PORT_TITLE(port)
    seen = {norm_terminal(t) for t in terminals_seen.dropna().unique().tolist()}
    return PORT_ENTRANT.get(p,"SIPG") if PORT_ENTRANT.get(p,"SIPG") in (seen | {"SIPG","HCT"}) else "SIPG"
def map_qtr_to_index(df):
    sub = df.loc[df["qtr"].notna(), ["qtr","t_index"]].drop_duplicates()
    dup = sub.groupby("qtr")["t_index"].nunique()
    if (dup>1).any():
        raise ValueError("Non-unique qtr->t_index mapping.")
    return dict(zip(sub["qtr"], sub["t_index"]))
def robust_meta(df):
    y = pd.to_numeric(df.get("year"), errors="coerce")
    return {"n_rows": int(len(df)), "n_ports": int(df["port"].nunique()) if "port" in df.columns else 0,
            "n_terminals": int(df["terminal"].nunique()) if "terminal" in df.columns else 0,
            "n_pre_synth": int(pd.to_numeric(df.get("is_synth_pre"), errors="coerce").fillna(0).sum()) if "is_synth_pre" in df.columns else 0,
            "year_min": int(y.dropna().min()) if pd.notna(y.dropna().min()) else None,
            "year_max": int(y.dropna().max()) if pd.notna(y.dropna().max()) else None,
            "n_year_missing": int(y.isna().sum()),
            "has_tau_comp": bool(df.get("tau_comp").notna().any()) if "tau_comp" in df.columns else False,
            "has_tau_priv": bool(df.get("tau_priv").notna().any()) if "tau_priv" in df.columns else False,
            "has_nyt_ok_comp": bool(df.get("nyt_ok_comp").notna().any()) if "nyt_ok_comp" in df.columns else False}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    inp, out_dir = find_input(STEP3_CANDIDATES)
    df = pd.read_csv(inp, sep="\t")
    need = {"port","terminal","year","quarter","qtr","t_index","Y"}
    miss = need - set(df.columns)
    if miss:
        sys.exit(f"Missing required columns: {sorted(miss)}")
    df1 = df.copy()
    df1["terminal_norm"] = df1["terminal"].apply(norm_terminal)
    pre = df1[df1["terminal_norm"] == ""].copy()
    post = df1[df1["terminal_norm"] != ""].copy()
    entrant_map = {p: expected_entrant_for_port(p, post.loc[post["port"]==p, "terminal_norm"]) for p in df1["port"].dropna().unique().tolist()}
    pre_dups = []
    for _, row in pre.iterrows():
        base = row[["port","year","quarter","qtr","t_index","Y"]].copy()
        r1 = base.copy(); r1["terminal"]="Legacy"; r1["is_synth_pre"]=1; pre_dups.append(r1)
        r2 = base.copy(); r2["terminal"]=entrant_map.get(row["port"],"SIPG"); r2["is_synth_pre"]=1; pre_dups.append(r2)
    pre_expanded = pd.DataFrame(pre_dups) if pre_dups else pd.DataFrame(columns=["port","year","quarter","qtr","t_index","Y","terminal","is_synth_pre"])
    post_keep = post.copy(); post_keep["terminal"] = post_keep["terminal_norm"]
    post_keep = post_keep[post_keep["terminal"].isin(KEEP_TERMS_CANON)].copy(); post_keep["is_synth_pre"]=0
    stacked = pd.concat([post_keep[["port","year","quarter","qtr","t_index","Y","terminal","is_synth_pre"]], pre_expanded], axis=0, ignore_index=True)
    final = stacked.sort_values(["port","terminal","year","quarter"], kind="mergesort").drop_duplicates(subset=["port","terminal","year","quarter"], keep="first")
    final["tau_comp"] = np.nan; final["tau_priv"] = np.nan; final["nyt_ok_comp"] = np.nan
    yml = load_yaml_optional(); events_used = {}
    if yml is not None:
        qmap = map_qtr_to_index(df1)
        events = (yml.get("events", {}) or {})
        comp = (events.get("competition", {}) or {})
        priv = (events.get("privatization", {}) or {})
        G_comp, G_priv = {}, {}
        for port, q in comp.items():
            if q and isinstance(q,str) and (q in qmap):
                G_comp[PORT_TITLE(port)] = qmap[q]
        for port, q in priv.items():
            if q and isinstance(q,str) and (q in qmap):
                G_priv[PORT_TITLE(port)] = qmap[q]
        events_used = {"G_comp": G_comp, "G_priv": G_priv}
        if G_comp:
            comp_cut = final["port"].map(PORT_TITLE).map(G_comp)
            m = comp_cut.notna(); final.loc[m, "tau_comp"] = final.loc[m,"t_index"].astype(int) - comp_cut[m].astype(int)
        if G_priv:
            priv_cut = final["port"].map(PORT_TITLE).map(G_priv)
            m2 = priv_cut.notna(); final.loc[m2, "tau_priv"] = final.loc[m2,"t_index"].astype(int) - priv_cut[m2].astype(int)
        if {"Haifa","Ashdod"}.issubset(set(G_comp.keys())):
            cutoff_map = {"Haifa": G_comp["Ashdod"], "Ashdod": G_comp["Haifa"]}
            cutoff = final["port"].map(PORT_TITLE).map(cutoff_map)
            m3 = cutoff.notna(); final.loc[m3, "nyt_ok_comp"] = (final.loc[m3,"t_index"].astype(int) < cutoff[m3].astype(int)).astype("int64")
    out_csv = out_dir / "04_panel_terminal_sharedpre_log.csv"
    final.to_csv(out_csv, index=False)
    meta = robust_meta(final)
    (out_dir / "04__meta_panel_terminal_sharedpre_log.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if args.debug:
        bundle_rows: List[pd.DataFrame] = []
        sample = df.head(300).copy(); sample["stage"] = "S0_raw_sample"; bundle_rows.append(sample[["stage","port","terminal","year","quarter","qtr","t_index","Y"]])
        p2 = pre_expanded.copy(); p2["stage"]="S2_pre_dups"; bundle_rows.append(p2[["stage","port","terminal","year","quarter","qtr","t_index","Y","is_synth_pre"]])
        k2 = post_keep.copy(); k2["stage"]="S3_post_kept"; bundle_rows.append(k2[["stage","port","terminal","year","quarter","qtr","t_index","Y","is_synth_pre"]])
        s2 = stacked.copy(); s2["stage"]="S4_stacked"; bundle_rows.append(s2[["stage","port","terminal","year","quarter","qtr","t_index","Y","is_synth_pre"]])
        f2 = final.copy(); f2["stage"]="S6_final"; bundle_rows.append(f2[["stage","port","terminal","year","quarter","qtr","t_index","Y","is_synth_pre"]])
        bundle = pd.concat(bundle_rows, axis=0, ignore_index=True)
        (out_dir / "04__debug_bundle.csv").write_text(bundle.to_csv(index=False), encoding="utf-8")
        report = {"summary_counts": {"raw_n": int(len(df)), "pre_n": int(len(pre)), "post_n": int(len(post)), "pre_dups_n": int(len(pre_expanded)), "post_kept_n": int(len(post_keep)), "stacked_n": int(len(stacked)), "final_n": int(len(final))}, "final_meta": meta}
        (out_dir / "04__debug_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[write] {out_csv}")
    print(f"[write] {out_dir / "_meta_panel_terminal_sharedpre_log.json"}")
    if args.debug:
        print(f"[write] {out_dir / "04__debug_bundle.csv"}")
        print(f"[write] {out_dir / "04__debug_report.json"}")
if __name__ == "__main__":
    main()
