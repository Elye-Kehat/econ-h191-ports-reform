#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_make_tables.py
------------------
Generates JOLE-style publication tables from Step-07 outputs.
"""

# ====================== USER TOGGLE =======================
OUTDIR = "Design/Output Data"
P_FOR_STARS = "p_wild"
DEC = 3
TABLE_NOTES = [
    "Outcome is ln(LP). Event-study uses not-yet-treated comparisons with terminal and calendar-quarter fixed effects.",
    "Baseline bin k=-1 is omitted (donut at event). Pre-trends p-value is a joint F-test over lead bins.",
    "Standard errors are clustered by port. Significance stars use the wild-cluster p-value when available (*** p<0.01, ** p<0.05, * p<0.10).",
]
# ==========================================================

import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

def ensure_outdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def latex_escape(s: str) -> str:
    return (s.replace("&", "\\&")
             .replace("%", "\\%")
             .replace("$", "\\$")
             .replace("#", "\\#")
             .replace("_", "\\_")
             .replace("{", "\\{")
             .replace("}", "\\}")
             .replace("~", "\\textasciitilde{}")
             .replace("^", "\\textasciicircum{}")
             .replace("\\", "\\textbackslash{}"))

def starify(pval: float) -> str:
    if pd.isna(pval):
        return ""
    if pval < 0.01: return "\\sym{***}"
    if pval < 0.05: return "\\sym{**}"
    if pval < 0.10: return "\\sym{*}"
    return ""

def fmt_coef_se(coef: float, se: float, pval: float, dec: int = 3) -> str:
    stars = starify(pval)
    c = f"{coef:.{dec}f}{stars}"
    s = f"({se:.{dec}f})"
    return c, s

def read_step7(outdir: str):
    main = pd.read_csv(Path(outdir) / "07_table_main.csv") if (Path(outdir)/"07_table_main.csv").exists() else None
    ent = pd.read_csv(Path(outdir) / "07_es_coeffs_entrant.csv") if (Path(outdir)/"07_es_coeffs_entrant.csv").exists() else None
    leg = pd.read_csv(Path(outdir) / "07_es_coeffs_legacy.csv") if (Path(outdir)/"07_es_coeffs_legacy.csv").exists() else None
    meta = None
    mp = Path(outdir) / "07__meta.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            meta = None
    return main, ent, leg, meta

def make_table1_main(main: pd.DataFrame, meta: dict, outdir: str):
    if main is None or main.empty:
        raise SystemExit("[08] Missing or empty 07_table_main.csv")

    role_order = ["entrant", "legacy"]
    df = main.copy()
    pcol = P_FOR_STARS if P_FOR_STARS in df.columns else ("p_crv" if "p_crv" in df.columns else None)
    if pcol is None:
        raise SystemExit("[08] Could not find p-value column in 07_table_main.csv")

    rows = []
    avgpost = {"row": "Average post"}
    se_row  = {"row": ""}
    post_window, clusters, N = "", "", ""
    for role in role_order:
        r = df[df["role"] == role].head(1)
        if r.empty:
            avgpost[role] = ""; se_row[role] = ""
        else:
            coef = float(r["coef"].iloc[0])
            se   = float(r["se"].iloc[0])
            p    = float(r[pcol].iloc[0]) if pcol in r.columns else np.nan
            c, s = fmt_coef_se(coef, se, p, DEC)
            avgpost[role] = c
            se_row[role]  = f"\\scriptsize {s}"
            post_window   = r.get("post_window", pd.Series([post_window])).iloc[0]
            clusters      = int(r.get("clusters", pd.Series([clusters])).iloc[0]) if "clusters" in r else clusters
            N             = int(r.get("N", pd.Series([N])).iloc[0]) if "N" in r else N
    rows += [avgpost, se_row]

    pre = {"row": "Pre-trends p-value (leads joint F)"}
    if meta and "pretrend_p" in meta:
        for role in role_order:
            val = meta["pretrend_p"].get(role, None)
            pre[role] = f"{val:.3f}" if (val is not None and not pd.isna(val)) else ""
    else:
        pre["entrant"] = ""; pre["legacy"] = ""
    rows.append(pre)

    rows.extend([
        {"row": "Terminal FE", "entrant": "Yes", "legacy": "Yes"},
        {"row": "Quarter FE",  "entrant": "Yes", "legacy": "Yes"},
        {"row": "Omitted bin", "entrant": "k = -1", "legacy": "k = -1"},
        {"row": "Post window", "entrant": str(post_window), "legacy": str(post_window)},
        {"row": "Observations","entrant": f"{N}", "legacy": f"{N}"},
        {"row": "Clusters",    "entrant": f"{clusters}", "legacy": f"{clusters}"},
    ])

    long = pd.DataFrame(rows)
    csv_path = Path(outdir) / "08_table1_main.csv"
    long.to_csv(csv_path, index=False)

    colspec = "lcc"
    head = (
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{Main Results — Average Post-Reform Effect on ln(LP)}\n"
        "\\label{tab:main}\n"
        "\\begin{tabular}{%s}\n\\toprule\n" % colspec +
        " & Entrant (pooled) & Legacy (pooled) \\\\n\\midrule\n"
    )
    body = ""
    for _, r in long.iterrows():
        left = latex_escape(str(r["row"]))
        c1 = r.get("entrant", "")
        c2 = r.get("legacy", "")
        body += f"{left} & {c1} & {c2} \\\\n"
    notes = (
        "\\midrule\n\\multicolumn{3}{p{0.9\\linewidth}}{\\footnotesize "
        + " ".join(latex_escape(x) for x in TABLE_NOTES)
        + " }\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        "\\newcommand{\\sym}[1]{\\ifmmode^{#1}\\else\\(^{#1}\\)\\fi}\n"
    )
    tex = head + body + notes
    (Path(outdir) / "08_table1_main.tex").write_text(tex, encoding="utf-8")

def make_table2_dynamic(ent: pd.DataFrame, leg: pd.DataFrame, outdir: str):
    if ent is None or ent.empty:
        raise SystemExit("[08] Missing or empty 07_es_coeffs_entrant.csv")
    if leg is None or leg.empty:
        raise SystemExit("[08] Missing or empty 07_es_coeffs_legacy.csv")

    def prep(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy().sort_values("k")
        d["beta_fmt"] = d["beta"].map(lambda x: f"{x:.{DEC}f}")
        d["se_fmt"]   = d["se"].map(lambda x: f"({x:.{DEC}f})")
        d["beta_se"]  = d["beta_fmt"] + " " + d["se_fmt"]
        return d[["k","beta_se","n_k"]]

    ent2 = prep(ent)
    leg2 = prep(leg)

    ent2.to_csv(Path(outdir)/"08_table2_dynamic_pooled_panelA_entrant.csv", index=False)
    leg2.to_csv(Path(outdir)/"08_table2_dynamic_pooled_panelB_legacy.csv", index=False)

    head = (
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{Event-Time Estimates — Pooled Entrant and Legacy}\n"
        "\\label{tab:dynamic}\n"
        "\\begin{tabular}{lcc}\n\\toprule\n"
        " & $\\beta$ (SE) & $N_k$ \\\\n\\midrule\n"
        "\\multicolumn{3}{l}{\\textit{Panel A: Entrant (pooled)}}\\\\n"
    )
    body = ""
    for _, r in ent2.iterrows():
        body += f"{int(r['k'])} & {r['beta_se']} & {int(r['n_k'])} \\\\n"
    body += "\\midrule\n\\multicolumn{3}{l}{\\textit{Panel B: Legacy (pooled)}}\\\\n"
    for _, r in leg2.iterrows():
        body += f"{int(r['k'])} & {r['beta_se']} & {int(r['n_k'])} \\\\n"

    notes = (
        "\\midrule\n\\multicolumn{3}{p{0.9\\linewidth}}{\\footnotesize "
        "Baseline k=$-1$ omitted. Coefficients from NYT design with terminal and quarter FE; SEs clustered by port. "
        "Bins are quarters relative to entry; tails may be sparse (see $N_k$). "
        "}\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    tex = head + body + notes
    (Path(outdir) / "08_table2_dynamic_pooled.tex").write_text(tex, encoding="utf-8")

def main():
    here = Path(__file__).resolve()
    try:
        os.chdir(here.parents[3])
    except Exception:
        os.chdir(here.parent)

    ensure_outdir(OUTDIR)
    main_df = ent_df = leg_df = meta = None
    # Read step-07 artifacts
    from pathlib import Path as P
    p_main = P(OUTDIR)/"07_table_main.csv"
    p_ent  = P(OUTDIR)/"07_es_coeffs_entrant.csv"
    p_leg  = P(OUTDIR)/"07_es_coeffs_legacy.csv"
    p_meta = P(OUTDIR)/"07__meta.json"
    if p_main.exists(): main_df = pd.read_csv(p_main)
    if p_ent.exists():  ent_df  = pd.read_csv(p_ent)
    if p_leg.exists():  leg_df  = pd.read_csv(p_leg)
    if p_meta.exists():
        try:
            meta = json.loads(p_meta.read_text(encoding="utf-8"))
        except Exception:
            meta = None

    make_table1_main(main_df, meta, OUTDIR)
    make_table2_dynamic(ent_df, leg_df, OUTDIR)

    meta_out = {
        "inputs": {
            "07_table_main.csv": Path(OUTDIR, "07_table_main.csv").exists(),
            "07_es_coeffs_entrant.csv": Path(OUTDIR, "07_es_coeffs_entrant.csv").exists(),
            "07_es_coeffs_legacy.csv": Path(OUTDIR, "07_es_coeffs_legacy.csv").exists(),
            "07__meta.json": Path(OUTDIR, "07__meta.json").exists(),
        },
        "outputs": [
            "08_table1_main.csv",
            "08_table1_main.tex",
            "08_table2_dynamic_pooled_panelA_entrant.csv",
            "08_table2_dynamic_pooled_panelB_legacy.csv",
            "08_table2_dynamic_pooled.tex",
        ],
        "p_for_stars": P_FOR_STARS,
        "decimals": DEC,
    }
    Path(OUTDIR, "08__meta_tables.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print("[08] Wrote JOLE-style tables to", OUTDIR)

if __name__ == "__main__":
    main()
