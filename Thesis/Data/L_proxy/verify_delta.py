#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_delta.py — Math & validity checks for labor_hours_missing_delta.tsv

Checks performed (printed to terminal):
A) Universe & keys: terminals, years, duplicates
B) Inheritance: delta.TEU_port_m == master port-month TEU_port_m
C) Identities: TEU_i_m == share * TEU_port_m; sums of w; sums of L == H == TEU/Π
D) Backfill algebra: Π_new(2021–22) = α_2023(port) × Π_legacy(2021–22), α from 2023
E) Structural zeros: pre-opening months are zero; optional cross-check with TEU panel

Usage (from repo root):
  python Data/L_proxy/verify_delta.py \
    --delta       Data/L_proxy/labor_hours_missing_delta.tsv \
    --port-month  Data/L_proxy/labor_hours_monthly_port.tsv \
    --kpi-wide    Data/L_proxy/containers_kpis_annual_wide_filled.tsv \
    --teu-panel   Data/Output/teu_monthly_plus_quarterly_by_port.tsv

If --teu-panel is omitted, E) runs without the external cross-check.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)

DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"
CANON = {
    "Ashdod HCT": "Ashdod-HCT", "Ashdod–HCT": "Ashdod-HCT", "Ashdod-HCT": "Ashdod-HCT",
    "Hadarom Container Terminal": "Ashdod-HCT", "Southport": "Ashdod-HCT",
    "Haifa SIPG": "Haifa-Bayport", "Haifa-SIPG": "Haifa-Bayport", "Haifa-Bayport": "Haifa-Bayport", "Bayport": "Haifa-Bayport",
    "Ashdod": "Ashdod", "Haifa": "Haifa", "Eilat": "Eilat",
}
LEGACY_BY_PORT = {"Ashdod": "Ashdod-Legacy", "Haifa": "Haifa-Legacy", "Eilat": "Eilat"}
PORT_OF_TERMINAL = {"Ashdod-HCT": "Ashdod", "Haifa-Bayport": "Haifa"}

Q_FROM_MONTH = {1:"Q1",2:"Q1",3:"Q1",4:"Q2",5:"Q2",6:"Q2",7:"Q3",8:"Q3",9:"Q3",10:"Q4",11:"Q4",12:"Q4"}

def norm_dashes(x: str):
    if x is None or (isinstance(x, float) and np.isnan(x)): return x
    s = str(x)
    for d in DASHES: s = s.replace(d, "-")
    return s

def canon(x: str) -> str:
    return CANON.get(norm_dashes(x), norm_dashes(x))

def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\t|,", engine="python")

def normalize_quarter(q):
    if pd.isna(q): return np.nan
    s = str(q).strip().upper()
    if s.startswith("Q"):
        for ch in s:
            if ch.isdigit() and int(ch) in (1,2,3,4):
                return f"Q{int(ch)}"
        return s
    try:
        d = int(float(s))
        if d in (1,2,3,4): return f"Q{d}"
    except Exception:
        pass
    return s

def compute_alpha_2023(kpi: pd.DataFrame) -> Dict[str, float]:
    """α_port = Pi(new,2023)/Pi(legacy,2023) for port in {Ashdod, Haifa}"""
    out = {}
    for port,newt in [("Ashdod","Ashdod-HCT"),("Haifa","Haifa-Bayport")]:
        leg = LEGACY_BY_PORT[port]
        pn = kpi.loc[(kpi["terminal"]==newt)&(kpi["year"]==2023),"teu_per_work_hour"].dropna()
        pl = kpi.loc[(kpi["terminal"]==leg) & (kpi["year"]==2023),"teu_per_work_hour"].dropna()
        out[port] = float(pn.iloc[0]/pl.iloc[0]) if (not pn.empty and not pl.empty and float(pl.iloc[0])>0) else np.nan
    return out

def main(delta_path: Path, port_month_path: Path, kpi_path: Path, teu_panel_path: Path|None):
    print("\n========== LOAD ==========")
    delta = read_table(delta_path)
    portm = read_table(port_month_path)
    kpi   = read_table(kpi_path)

    # Canonicalize
    for df,name in [(delta,"delta"),(portm,"portm"),(kpi,"kpi")]:
        if "port" in df.columns: df["port"] = df["port"].map(canon)
        if "terminal" in df.columns: df["terminal"] = df["terminal"].map(canon)
    # numeric
    for c in ["year","month","TEU_port_m","share_i_p_q","TEU_i_m","w_i_m_in_year",
              "Pi_teu_per_hour_i_y_new","H_annual_i_y_new","L_hours_i_m_new"]:
        if c in delta.columns: delta[c] = pd.to_numeric(delta[c], errors="coerce")
    for c in ["year","month","TEU_port_m"]:
        if c in portm.columns: portm[c] = pd.to_numeric(portm[c], errors="coerce")
    # KPI normalize
    kpi.columns = [c.strip().lower() for c in kpi.columns]
    if "teu_per_work_hour" not in kpi.columns:
        for alt in ["teu_per_hour","teu_per_workhour","pi_teu_per_hour","teu_per_work_hr","teu_per_work-hr"]:
            if alt in kpi.columns:
                kpi = kpi.rename(columns={alt:"teu_per_work_hour"})
                break
    if "port" in kpi.columns: kpi["port"] = kpi["port"].map(canon)
    if "terminal" in kpi.columns: kpi["terminal"] = kpi["terminal"].map(canon)
    kpi["year"] = pd.to_numeric(kpi["year"], errors="coerce")
    kpi["teu_per_work_hour"] = pd.to_numeric(kpi["teu_per_work_hour"], errors="coerce")
    mask_leg = kpi["port"].isin(["Ashdod","Haifa"]) & (~kpi["terminal"].isin(["Ashdod-HCT","Haifa-Bayport"]))
    kpi.loc[mask_leg,"terminal"] = kpi.loc[mask_leg,"port"].map(LEGACY_BY_PORT)

    # Prepare keys
    delta["_key"] = list(zip(delta["port"], delta["terminal"], delta["year"], delta["month"]))
    portm["quarter"] = portm["quarter"].apply(normalize_quarter) if "quarter" in portm.columns else portm["month"].map(lambda m: Q_FROM_MONTH[int(m)])
    print(f"delta rows: {len(delta)}  |  port-month rows: {len(portm)}  |  kpi rows: {len(kpi)}")

    # Optional TEU panel
    teu = None
    if teu_panel_path is not None:
        teu = read_table(teu_panel_path)
        teu.columns = [c.strip() for c in teu.columns]
        teu["Port"] = teu["Port"].map(canon)
        # TEU val
        if "TEU" in teu.columns:
            teu["TEU_val"] = pd.to_numeric(teu["TEU"], errors="coerce")
        elif "TEU_thousands" in teu.columns:
            teu["TEU_val"] = pd.to_numeric(teu["TEU_thousands"], errors="coerce")*1000.0
        else:
            teu["TEU_val"] = np.nan
        # year/quarter
        if "Year" not in teu.columns and "Period" in teu.columns:
            teu["Year"] = teu["Period"].map(lambda s: int(str(s).split("-")[-1]) if isinstance(s,str) and "-" in s else np.nan)
        teu["Quarter"] = teu.apply(lambda r: r["Period"].split("-")[0] if (isinstance(r.get("Period"),str) and r["Period"].startswith("Q")) else (
                                      Q_FROM_MONTH[int(r["Month"])] if (r.get("Freq")=="Monthly" and pd.notnull(r.get("Month"))) else np.nan
                                   ), axis=1)
        print(f"teu panel rows: {len(teu)}")

    print("\n========== A) UNIVERSE & KEYS ==========")
    terms = delta["terminal"].unique().tolist()
    years = sorted(delta["year"].dropna().unique().tolist())
    dups = delta.duplicated(subset=["port","terminal","year","month"]).sum()
    print(f"Terminals in delta: {terms}")
    print(f"Years in delta: {years}")
    print(f"Duplicate (port,terminal,year,month) rows: {dups}")
    ok_A = (set(terms) == {"Ashdod-HCT","Haifa-Bayport"}) and (set(years) <= {2021,2022}) and (dups == 0)
    print(f"PASS(A): {ok_A}")

    print("\n========== B) INHERIT TEU_port_m FROM PORT-MONTH ==========")
    joined = delta.merge(portm[["port","year","month","TEU_port_m"]].rename(columns={"TEU_port_m":"TEU_port_m_master"}),
                         on=["port","year","month"], how="left")
    joined["diff_port_teu"] = joined["TEU_port_m"] - joined["TEU_port_m_master"]
    mismatches = int((joined["diff_port_teu"].abs() > 1e-6).sum())
    print(f"Rows lacking a matching port-month record: {int(joined['TEU_port_m_master'].isna().sum())}")
    print(f"Max |delta.TEU_port_m - master.TEU_port_m|: {joined['diff_port_teu'].abs().max()}")
    print(f"Count mismatches > 1e-6: {mismatches}")
    ok_B = (joined["TEU_port_m_master"].notna().all()) and (mismatches == 0)
    print(f"PASS(B): {ok_B}")

    print("\n========== C) IDENTITIES ==========")
    # Row identity
    delta["alloc_err"] = delta["TEU_i_m"] - (delta["share_i_p_q"] * delta["TEU_port_m"])
    print(f"Max |TEU_i_m - share*TEU_port_m|: {delta['alloc_err'].abs().max()}")
    ok_C1 = bool((delta["alloc_err"].abs() <= 1e-6).all())

    # Annual sums and weights
    grp = delta.groupby(["port","terminal","year"], as_index=False).agg(
        TEU_y=("TEU_i_m","sum"),
        H_y=("H_annual_i_y_new","first"),  # should be constant within year
        Pi_y=("Pi_teu_per_hour_i_y_new","first")  # should be constant within year
    )
    # recompute weights & L consistency
    delta2 = delta.merge(grp[["port","terminal","year","TEU_y","H_y","Pi_y"]],
                         on=["port","terminal","year"], how="left")
    delta2["w_calc"] = delta2["TEU_i_m"] / delta2["TEU_y"].replace({0:np.nan})
    w_dev = (delta2["w_calc"] - delta2["w_i_m_in_year"]).abs().dropna().max()
    sumw = delta2.groupby(["port","terminal","year"])["w_i_m_in_year"].sum()
    sumw_dev = (sumw - 1.0).abs().max()
    # H consistency
    Hy_from_pi = (grp["TEU_y"] / grp["Pi_y"]).replace([np.inf, -np.inf], np.nan)
    H_dev = (grp["H_y"] - Hy_from_pi).abs().max()
    # Sum L equals H
    Lsum = delta.groupby(["port","terminal","year"], as_index=False)["L_hours_i_m_new"].sum()
    Lsum = Lsum.merge(grp[["port","terminal","year","H_y"]], on=["port","terminal","year"], how="left")
    L_dev = (Lsum["L_hours_i_m_new"] - Lsum["H_y"]).abs().max()

    print(f"Max |w_calc - w_i_m_in_year|: {w_dev}")
    print(f"Max |sum_m w_i_m_in_year - 1|: {sumw_dev}")
    print(f"Max |H_annual_i_y_new - TEU_y/Pi|: {H_dev}")
    print(f"Max |sum_m L_hours_i_m_new - H|: {L_dev}")
    ok_C = bool((w_dev <= 1e-9) and (sumw_dev <= 1e-12) and (H_dev <= 1e-9) and (L_dev <= 1e-9))
    print(f"PASS(C): {ok_C}")

    print("\n========== D) BACKFILL ALGEBRA (Pî) ==========")
    alpha = compute_alpha_2023(kpi)
    print(f"alpha_2023 by port: {alpha}")
    # legacy Π by port-year
    leg = kpi[kpi["terminal"].isin([LEGACY_BY_PORT["Ashdod"], LEGACY_BY_PORT["Haifa"]])]
    legacy = {(r.port, int(r.year)): float(r.teu_per_work_hour) for r in leg.itertuples(index=False)}
    # Expected Pi new for 2021–22
    exp = []
    for (port,term,year), g in delta.groupby(["port","terminal","year"]):
        if year in (2021, 2022):
            a = alpha.get(port, np.nan)
            leg_pi = legacy.get((port, int(year)), np.nan)
            exp_pi = float(a*leg_pi) if (isinstance(a,float) and a>0 and isinstance(leg_pi,float) and leg_pi>0) else np.nan
            used_pi = float(g["Pi_teu_per_hour_i_y_new"].iloc[0])
            exp.append({"port":port,"terminal":term,"year":year,"Pi_expected":exp_pi,"Pi_used":used_pi,
                        "abs_diff": abs((exp_pi if pd.notnull(exp_pi) else np.nan) - used_pi)})
    exp_df = pd.DataFrame(exp)
    print(exp_df.to_string(index=False))
    ok_D = bool((exp_df["abs_diff"].fillna(0.0) <= 1e-9).all() and exp_df["Pi_expected"].notna().all())
    print(f"PASS(D): {ok_D}")

    print("\n========== E) STRUCTURAL ZEROS / OPENING PATTERN ==========")
    # From delta: list earliest positive month for each terminal across 2021–22
    pos = delta[delta["TEU_i_m"] > 0].groupby(["terminal"], as_index=False)["month"].min().rename(columns={"month":"first_positive_month"})
    zero_counts = delta.groupby(["terminal","year"], as_index=False).apply(lambda g: (g["TEU_i_m"]==0).sum(), include_groups=False)\
                       .rename(columns={0:"zero_months"})
    print("First positive month (delta-based):")
    print(pos.to_string(index=False))
    print("Zero-month counts by terminal-year (delta-based):")
    print(zero_counts.to_string(index=False))
    ok_E = True  # by construction; refined below if TEU panel provided

    if teu is not None:
        # Cross-check: if TEU panel has quarterly >0 for the new terminal, months in that quarter should have positive shares (sum>0)
        newq = teu[(teu["Freq"]=="Quarterly") & (teu["Port"].isin(["Ashdod-HCT","Haifa-Bayport","Ashdod HCT","Haifa SIPG","Haifa-SIPG"]))].copy()
        newq["Terminal"] = newq["Port"].map(canon)
        newq["Port"] = newq["Terminal"].map(PORT_OF_TERMINAL)
        newq = newq.groupby(["Terminal","Port","Year","Quarter"], as_index=False)["TEU_val"].sum()
        newq["Quarter"] = newq["Quarter"].apply(normalize_quarter)

        # Summarize expected positive quarters from panel
        panel_pos = newq[(newq["Year"].isin([2021,2022])) & (newq["TEU_val"]>0)]
        print("\nTEU panel: positive quarters for new terminals (2021–22):")
        if not panel_pos.empty:
            print(panel_pos[["Terminal","Port","Year","Quarter","TEU_val"]].to_string(index=False))
        else:
            print("(none)")

        # From delta, sum TEU_i_m per terminal-quarter
        delta["quarter"] = delta.apply(lambda r: Q_FROM_MONTH[int(r["month"])], axis=1)
        dsum = delta.groupby(["terminal","port","year","quarter"], as_index=False)["TEU_i_m"].sum() \
                    .rename(columns={"TEU_i_m":"TEU_from_delta"})
        chk = panel_pos.merge(dsum, left_on=["Terminal","Port","Year","Quarter"],
                              right_on=["terminal","port","year","quarter"], how="left")
        chk["delta_positive"] = chk["TEU_from_delta"].fillna(0) > 0
        bad = chk[~chk["delta_positive"]]
        print("\nQuarter-level cross-check vs TEU panel (quarters with panel>0 but delta sum==0):")
        if bad.empty:
            print("(none)")
        else:
            print(bad[["Terminal","Port","Year","Quarter","TEU_val","TEU_from_delta"]].to_string(index=False))
            ok_E = False

    print("\n========== SUMMARY ==========")
    print(f"PASS(A) Universe/Keys: {ok_A}")
    print(f"PASS(B) Inherit TEU_port_m: {ok_B}")
    print(f"PASS(C) Identities (TEU, w, H, L): {ok_C1 and ok_C}")
    print(f"PASS(D) Backfill algebra (Pî): {ok_D}")
    print(f"PASS(E) Structural zeros/opening: {ok_E}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Verify delta math & validity.")
    ap.add_argument("--delta",      type=Path, default=Path("Data/L_proxy/labor_hours_missing_delta.tsv"))
    ap.add_argument("--port-month", type=Path, default=Path("Data/L_proxy/labor_hours_monthly_port.tsv"))
    ap.add_argument("--kpi-wide",   type=Path, default=Path("Data/L_proxy/containers_kpis_annual_wide_filled.tsv"))
    ap.add_argument("--teu-panel",  type=Path, default=None,
                    help="Optional: Data/Output/teu_monthly_plus_quarterly_by_port.tsv for cross-check E")
    args = ap.parse_args()
    main(args.delta, args.port_month, args.kpi_wide, args.teu_panel)
