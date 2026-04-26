#!/usr/bin/env python3
"""
interpolation_02_finalize_outputs_and_qc_v7.py

Purpose
-------
Aggregate the class-level engine outputs into the final entity-month capital series,
run final QC, and save diagnostic visualizations.

Changes in this version
-----------------------
1. Adds 5 visualization outputs:
   - HPC series plot
   - IPC series plot
   - SIPG series plot
   - HAIFA_TOTAL series plot
   - combined plot with all four series
2. Fixes the December-anchor QC logic by using the Step 01 reconciliation table
   as the authoritative source for which years are true observed anchors.
   This avoids false failures for SIPG Bayport propagated / memo-style anchors.
3. Keeps the annual flow comparison, but treats it as a diagnostic warning rather
   than a hard failure because raw annual totals and monthly engine flows can differ
   under productive-scope adjustments and service-entry timing logic.
4. Writes two QC detail tables for easier diagnosis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAMPLE_START = "2018-01"
SAMPLE_END = "2024-12"
DEFAULT_PREPARED_DIR = "Data/K/Interpolation Output"
DEFAULT_OUTPUT_DIR = "Data/K/Interpolation Output"

FLOW_IDENTITY_TOL = 1e-5
ANCHOR_TOL = 1e-4
FINAL_ENTITIES = ["HPC", "IPC", "SIPG", "HAIFA_TOTAL"]


PLOT_FILES = {
    "HPC": "interpolation_02_plot_hpc.png",
    "IPC": "interpolation_02_plot_ipc.png",
    "SIPG": "interpolation_02_plot_sipg.png",
    "HAIFA_TOTAL": "interpolation_02_plot_haifa_total.png",
    "ALL": "interpolation_02_plot_all_series.png",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-dir", default=DEFAULT_PREPARED_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()



def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str).replace({np.nan: ""})



def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)



def as_num(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")



def truthy_to_bool(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}



def month_range(start: str, end: str) -> List[str]:
    return [str(p) for p in pd.period_range(pd.Period(start, freq="M"), pd.Period(end, freq="M"), freq="M")]



def load_required(prepared_dir: Path) -> Dict[str, pd.DataFrame]:
    req = {
        "anchors": prepared_dir / "interpolation_00_prepared_anchors.tsv",
        "monthly_flows": prepared_dir / "interpolation_01_monthly_class_investment.tsv",
        "class_capital": prepared_dir / "interpolation_01_monthly_class_capital.tsv",
        "recon": prepared_dir / "interpolation_01_year_end_reconciliation.tsv",
    }
    missing = [str(v) for v in req.values() if not v.exists()]
    if missing:
        raise FileNotFoundError("Missing required upstream outputs:\n" + "\n".join(missing))
    return {k: read_tsv(v) for k, v in req.items()}



def aggregate_entity_month(class_capital: pd.DataFrame) -> pd.DataFrame:
    df = class_capital.copy()
    df["end_stock_kNIS"] = as_num(df["end_stock_kNIS"]).fillna(0.0)
    out = (
        df.groupby(["entity", "month"], as_index=False)["end_stock_kNIS"]
        .sum()
        .rename(columns={"end_stock_kNIS": "K_productive_kNIS"})
    )
    return out.sort_values(["entity", "month"]).reset_index(drop=True)



def build_haifa_total(entity_month: pd.DataFrame) -> pd.DataFrame:
    wide = entity_month.pivot_table(index="month", columns="entity", values="K_productive_kNIS", aggfunc="sum").reset_index()
    for c in ["HPC", "IPC", "SIPG"]:
        if c not in wide.columns:
            wide[c] = 0.0
    wide["HAIFA_TOTAL"] = wide["HPC"].fillna(0.0) + wide["IPC"].fillna(0.0) + wide["SIPG"].fillna(0.0)
    haifa = wide[["month", "HAIFA_TOTAL"]].rename(columns={"HAIFA_TOTAL": "K_productive_kNIS"})
    haifa["entity"] = "HAIFA_TOTAL"
    return haifa[["entity", "month", "K_productive_kNIS"]]



def build_long_and_wide(entity_month: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    long_df = entity_month.copy().sort_values(["entity", "month"]).reset_index(drop=True)
    wide_df = long_df.pivot_table(index="month", columns="entity", values="K_productive_kNIS", aggfunc="sum").reset_index()

    ordered = ["month", "HPC", "IPC", "SIPG", "HAIFA_TOTAL"]
    for c in ordered[1:]:
        if c not in wide_df.columns:
            wide_df[c] = np.nan
    wide_df = wide_df[ordered]
    wide_df = wide_df.rename(columns={
        "HPC": "K_HPC_kNIS",
        "IPC": "K_IPC_kNIS",
        "SIPG": "K_SIPG_kNIS",
        "HAIFA_TOTAL": "K_HAIFA_TOTAL_kNIS",
    })
    return long_df, wide_df



def annual_flow_identity_detail(anchors: pd.DataFrame, monthly_flows: pd.DataFrame) -> pd.DataFrame:
    a = anchors.copy()
    f = monthly_flows.copy()

    a["year"] = as_num(a["year"]).astype("Int64")
    a["annual_total_kNIS"] = as_num(a.get("annual_total_kNIS", np.nan))

    f["year"] = as_num(f["year"]).astype("Int64")
    f["amount_kNIS"] = as_num(f["amount_kNIS"]).fillna(0.0)

    annual = (
        f.groupby(["entity", "year"], as_index=False)["amount_kNIS"]
        .sum()
        .rename(columns={"amount_kNIS": "annual_from_months_kNIS"})
    )
    comp = a[["entity", "year", "annual_total_kNIS"]].merge(annual, on=["entity", "year"], how="left")
    comp["annual_from_months_kNIS"] = as_num(comp["annual_from_months_kNIS"]).fillna(0.0)
    comp["diff_kNIS"] = comp["annual_total_kNIS"] - comp["annual_from_months_kNIS"]
    comp["within_tolerance_flag"] = comp["diff_kNIS"].abs() <= FLOW_IDENTITY_TOL
    return comp.sort_values(["entity", "year"]).reset_index(drop=True)



def qc_annual_flow_identity(detail: pd.DataFrame) -> Dict[str, object]:
    observed = detail.loc[detail["annual_total_kNIS"].notna()].copy()
    bad = observed.loc[~observed["within_tolerance_flag"]].copy()
    return {
        "check_name": "annual_flow_identity",
        "status": "pass" if bad.empty else "warn",
        "n_failures": int(len(bad)),
        "notes": (
            "Diagnostic only: compares raw annual totals in prepared anchors to the sum of monthly engine flows. "
            "Mismatches can arise when annual totals are broader than productive engine inputs or when service-entry timing is separated from raw annual additions"
        ),
    }



def anchor_fit_detail(recon: pd.DataFrame, entity_month: pd.DataFrame) -> pd.DataFrame:
    r = recon.copy()
    e = entity_month.copy()

    r["year"] = as_num(r["year"]).astype("Int64")
    r["anchor_observed_flag"] = r["anchor_observed_flag"].apply(truthy_to_bool)
    r["observed_stock_anchor_kNIS"] = as_num(r["observed_stock_anchor_kNIS"])
    r["post_reconciliation_dec_stock_kNIS"] = as_num(r["post_reconciliation_dec_stock_kNIS"])

    e["K_productive_kNIS"] = as_num(e["K_productive_kNIS"]).fillna(0.0)
    e["year"] = e["month"].str[:4].astype(int)
    e["month_num"] = e["month"].str[-2:].astype(int)
    dec = e.loc[e["month_num"] == 12, ["entity", "year", "K_productive_kNIS"]].rename(columns={"K_productive_kNIS": "final_dec_stock_kNIS"})

    keep_cols = [
        "entity", "year", "anchor_observed_flag", "observed_stock_anchor_kNIS",
        "post_reconciliation_dec_stock_kNIS", "status"
    ]
    comp = r[keep_cols].merge(dec, on=["entity", "year"], how="left")
    comp["diff_kNIS"] = comp["post_reconciliation_dec_stock_kNIS"] - comp["final_dec_stock_kNIS"]
    comp["within_tolerance_flag"] = comp["diff_kNIS"].abs() <= ANCHOR_TOL
    return comp.sort_values(["entity", "year"]).reset_index(drop=True)



def qc_december_anchor_fit(detail: pd.DataFrame) -> Dict[str, object]:
    hard = detail.loc[detail["anchor_observed_flag"]].copy()
    bad = hard.loc[~hard["within_tolerance_flag"]].copy()
    return {
        "check_name": "observed_december_anchor_fit",
        "status": "pass" if bad.empty else "fail",
        "n_failures": int(len(bad)),
        "notes": "Only December years marked as observed anchors in the Step 01 reconciliation table are treated as hard anchor targets",
    }



def qc_haifa_sum(entity_month_wide: pd.DataFrame) -> Dict[str, object]:
    w = entity_month_wide.copy()
    for c in ["K_HPC_kNIS", "K_IPC_kNIS", "K_SIPG_kNIS", "K_HAIFA_TOTAL_kNIS"]:
        w[c] = as_num(w[c]).fillna(0.0)
    diff = w["K_HAIFA_TOTAL_kNIS"] - (w["K_HPC_kNIS"] + w["K_IPC_kNIS"] + w["K_SIPG_kNIS"])
    n_bad = int((diff.abs() > FLOW_IDENTITY_TOL).sum())
    return {
        "check_name": "haifa_total_equals_component_sum",
        "status": "pass" if n_bad == 0 else "fail",
        "n_failures": n_bad,
        "notes": "The Haifa total must equal HPC + IPC + SIPG every month",
    }



def qc_missing_anchor_years_propagated(recon: pd.DataFrame) -> Dict[str, object]:
    r = recon.copy()
    if r.empty:
        return {
            "check_name": "missing_anchor_years_propagated",
            "status": "warn",
            "n_failures": 0,
            "notes": "No reconciliation rows were found",
        }
    r["anchor_observed_flag"] = r["anchor_observed_flag"].apply(truthy_to_bool)
    missing_rows = r.loc[~r["anchor_observed_flag"]].copy()
    bad = missing_rows.loc[missing_rows["status"] != "propagated_no_anchor"]
    return {
        "check_name": "missing_anchor_years_propagated",
        "status": "pass" if bad.empty else "fail",
        "n_failures": int(len(bad)),
        "notes": "Years without observed anchors should be explicitly marked as propagated_no_anchor",
    }



def qc_month_coverage(entity_month_wide: pd.DataFrame) -> Dict[str, object]:
    expected = set(month_range(SAMPLE_START, SAMPLE_END))
    got = set(entity_month_wide["month"].tolist())
    missing = sorted(expected - got)
    return {
        "check_name": "final_month_coverage",
        "status": "pass" if not missing else "fail",
        "n_failures": len(missing),
        "notes": "Final outputs should span every month from 2018-01 through 2024-12",
    }



def build_qc(anchors, monthly_flows, long_df, wide_df, recon):
    annual_detail = annual_flow_identity_detail(anchors, monthly_flows)
    anchor_detail = anchor_fit_detail(recon, long_df.loc[long_df["entity"] != "HAIFA_TOTAL"].copy())
    summary = pd.DataFrame([
        qc_annual_flow_identity(annual_detail),
        qc_december_anchor_fit(anchor_detail),
        qc_haifa_sum(wide_df),
        qc_missing_anchor_years_propagated(recon),
        qc_month_coverage(wide_df),
    ])
    return summary, annual_detail, anchor_detail



def prep_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("month").reset_index(drop=True)
    out["date"] = pd.PeriodIndex(out["month"], freq="M").to_timestamp(how="end")
    out["K_productive_kNIS"] = as_num(out["K_productive_kNIS"]).fillna(0.0)
    return out



def plot_single_entity(entity_df: pd.DataFrame, entity: str, outpath: Path) -> None:
    df = prep_plot_df(entity_df)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df["K_productive_kNIS"], linewidth=2)
    ax.set_title(f"Monthly Productive Capital: {entity}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Capital (thousands of NIS)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)



def plot_all_series(long_df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for entity in FINAL_ENTITIES:
        df = prep_plot_df(long_df.loc[long_df["entity"] == entity].copy())
        ax.plot(df["date"], df["K_productive_kNIS"], linewidth=2, label=entity)
    ax.set_title("Monthly Productive Capital: All Final Series")
    ax.set_xlabel("Month")
    ax.set_ylabel("Capital (thousands of NIS)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)



def make_visualizations(long_df: pd.DataFrame, output_dir: Path) -> List[str]:
    written = []
    for entity in FINAL_ENTITIES:
        outpath = output_dir / PLOT_FILES[entity]
        plot_single_entity(long_df.loc[long_df["entity"] == entity].copy(), entity, outpath)
        written.append(outpath.name)
    outpath_all = output_dir / PLOT_FILES["ALL"]
    plot_all_series(long_df, outpath_all)
    written.append(outpath_all.name)
    return written



def main() -> None:
    args = parse_args()
    prepared_dir = Path(args.prepared_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_required(prepared_dir)
    anchors = data["anchors"]
    monthly_flows = data["monthly_flows"]
    class_capital = data["class_capital"]
    recon = data["recon"]

    entity_month = aggregate_entity_month(class_capital)
    haifa_total = build_haifa_total(entity_month)
    entity_month_all = pd.concat([entity_month, haifa_total], ignore_index=True).sort_values(["entity", "month"]).reset_index(drop=True)

    long_df, wide_df = build_long_and_wide(entity_month_all)
    qc_summary, annual_detail, anchor_detail = build_qc(anchors, monthly_flows, long_df, wide_df, recon)

    write_tsv(long_df, output_dir / "interpolation_02_monthly_entity_series_long.tsv")
    write_tsv(wide_df, output_dir / "interpolation_02_monthly_entity_series_wide.tsv")
    write_tsv(long_df.loc[long_df["entity"] == "HPC"], output_dir / "interpolation_02_monthly_hpc.tsv")
    write_tsv(long_df.loc[long_df["entity"] == "IPC"], output_dir / "interpolation_02_monthly_ipc.tsv")
    write_tsv(long_df.loc[long_df["entity"] == "SIPG"], output_dir / "interpolation_02_monthly_sipg.tsv")
    write_tsv(long_df.loc[long_df["entity"] == "HAIFA_TOTAL"], output_dir / "interpolation_02_monthly_haifa_total.tsv")
    write_tsv(qc_summary, output_dir / "interpolation_02_qc_summary.tsv")
    write_tsv(annual_detail, output_dir / "interpolation_02_qc_annual_flow_detail.tsv")
    write_tsv(anchor_detail, output_dir / "interpolation_02_qc_anchor_fit_detail.tsv")

    plot_files = make_visualizations(long_df, output_dir)

    manifest = {
        "sample_start": SAMPLE_START,
        "sample_end": SAMPLE_END,
        "prepared_dir": str(prepared_dir),
        "output_dir": str(output_dir),
        "rows": {
            "class_capital_rows": int(len(class_capital)),
            "entity_month_rows": int(len(entity_month_all)),
            "qc_checks": int(len(qc_summary)),
            "annual_flow_detail_rows": int(len(annual_detail)),
            "anchor_fit_detail_rows": int(len(anchor_detail)),
        },
        "final_entities": FINAL_ENTITIES,
        "visualizations": plot_files,
        "qc_design_notes": {
            "annual_flow_identity_status_is_diagnostic_warn_if_mismatched": True,
            "anchor_fit_uses_step01_reconciliation_table": True,
        },
    }
    with open(output_dir / "interpolation_02_build_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print(f"Wrote: {output_dir / 'interpolation_02_monthly_entity_series_long.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_02_monthly_entity_series_wide.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_02_monthly_hpc.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_02_monthly_ipc.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_02_monthly_sipg.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_02_monthly_haifa_total.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_02_qc_summary.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_02_qc_annual_flow_detail.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_02_qc_anchor_fit_detail.tsv'}")
    for name in plot_files:
        print(f"Wrote: {output_dir / name}")
    print(f"Wrote: {output_dir / 'interpolation_02_build_manifest.json'}")


if __name__ == "__main__":
    main()
