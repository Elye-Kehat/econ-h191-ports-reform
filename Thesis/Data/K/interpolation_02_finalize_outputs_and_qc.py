#!/usr/bin/env python3
"""
interpolation_02_finalize_outputs_and_qc.py

Purpose
-------
Aggregate the class-level engine outputs into the final entity-month capital series and run final QC.

Main tasks
----------
1. Load interpolation_00 and interpolation_01 outputs.
2. Aggregate class capital to entity-month capital.
3. Construct the Haifa total as HPC + IPC + SIPG.
4. Save the final four monthly series in long and wide form.
5. Run final QC checks against annual investment totals and observed stock anchors.
6. Write a concise QC summary and build manifest.

Outputs
-------
interpolation_02_monthly_entity_series_long.tsv
interpolation_02_monthly_entity_series_wide.tsv
interpolation_02_monthly_hpc.tsv
interpolation_02_monthly_ipc.tsv
interpolation_02_monthly_sipg.tsv
interpolation_02_monthly_haifa_total.tsv
interpolation_02_qc_summary.tsv
interpolation_02_build_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


SAMPLE_START = "2018-01"
SAMPLE_END = "2024-12"

DEFAULT_PREPARED_DIR = "Data/K/Interpolation Output"
DEFAULT_OUTPUT_DIR = "Data/K/Interpolation Output"

IDENTITY_TOL = 1e-6
ANCHOR_TOL = 1e-4


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
    out = {k: read_tsv(v) for k, v in req.items()}
    return out


def aggregate_entity_month(class_capital: pd.DataFrame) -> pd.DataFrame:
    df = class_capital.copy()
    df["end_stock_kNIS"] = as_num(df["end_stock_kNIS"]).fillna(0.0)
    out = (
        df.groupby(["entity", "month"], as_index=False)["end_stock_kNIS"]
        .sum()
        .rename(columns={"end_stock_kNIS": "K_productive_kNIS"})
    )
    out = out.sort_values(["entity", "month"]).reset_index(drop=True)
    return out


def build_haifa_total(entity_month: pd.DataFrame) -> pd.DataFrame:
    wide = entity_month.pivot_table(index="month", columns="entity", values="K_productive_kNIS", aggfunc="sum").reset_index()
    for c in ["HPC", "IPC", "SIPG"]:
        if c not in wide.columns:
            wide[c] = 0.0
    wide["HAIFA_TOTAL"] = wide["HPC"].fillna(0.0) + wide["IPC"].fillna(0.0) + wide["SIPG"].fillna(0.0)
    haifa = wide[["month", "HAIFA_TOTAL"]].rename(columns={"HAIFA_TOTAL": "K_productive_kNIS"})
    haifa["entity"] = "HAIFA_TOTAL"
    return haifa[["entity", "month", "K_productive_kNIS"]]


def build_long_and_wide(entity_month: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_df = entity_month.copy().sort_values(["entity", "month"]).reset_index(drop=True)
    wide_df = long_df.pivot_table(index="month", columns="entity", values="K_productive_kNIS", aggfunc="sum").reset_index()
    # Stable column order
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


def qc_annual_flow_identity(anchors: pd.DataFrame, monthly_flows: pd.DataFrame) -> Dict[str, object]:
    a = anchors.copy()
    f = monthly_flows.copy()
    a["year"] = as_num(a["year"]).astype("Int64")
    a["annual_total_kNIS"] = as_num(a["annual_total_kNIS"])
    f["year"] = as_num(f["year"]).astype("Int64")
    f["amount_kNIS"] = as_num(f["amount_kNIS"]).fillna(0.0)

    annual = f.groupby(["entity", "year"], as_index=False)["amount_kNIS"].sum().rename(columns={"amount_kNIS": "annual_from_months_kNIS"})
    comp = a[["entity", "year", "annual_total_kNIS"]].merge(annual, on=["entity", "year"], how="left")
    comp["annual_from_months_kNIS"] = comp["annual_from_months_kNIS"].fillna(0.0)
    comp["diff_kNIS"] = comp["annual_total_kNIS"] - comp["annual_from_months_kNIS"]

    bad = comp.loc[comp["annual_total_kNIS"].notna() & (comp["diff_kNIS"].abs() > IDENTITY_TOL)]
    return {
        "check_name": "annual_flow_identity",
        "status": "pass" if bad.empty else "fail",
        "n_failures": int(len(bad)),
        "notes": "Annual investment totals must equal the sum of monthly flows by entity-year",
    }


def qc_december_anchor_fit(anchors: pd.DataFrame, entity_month: pd.DataFrame) -> Dict[str, object]:
    a = anchors.copy()
    e = entity_month.copy()
    a["year"] = as_num(a["year"]).astype("Int64")
    a["stock_anchor_dec_kNIS"] = as_num(a["stock_anchor_dec_kNIS"])
    e["K_productive_kNIS"] = as_num(e["K_productive_kNIS"]).fillna(0.0)
    e["year"] = e["month"].str[:4].astype(int)
    e["month_num"] = e["month"].str[-2:].astype(int)
    dec = e.loc[e["month_num"] == 12, ["entity", "year", "K_productive_kNIS"]].rename(columns={"K_productive_kNIS": "dec_stock_kNIS"})

    comp = a[["entity", "year", "stock_anchor_dec_kNIS"]].merge(dec, on=["entity", "year"], how="left")
    comp["diff_kNIS"] = comp["stock_anchor_dec_kNIS"] - comp["dec_stock_kNIS"]
    observed = comp.loc[comp["stock_anchor_dec_kNIS"].notna()].copy()
    bad = observed.loc[observed["diff_kNIS"].abs() > ANCHOR_TOL]

    return {
        "check_name": "observed_december_anchor_fit",
        "status": "pass" if bad.empty else "fail",
        "n_failures": int(len(bad)),
        "notes": "Observed December stock anchors should match the post-reconciliation December stock",
    }


def qc_haifa_sum(entity_month_long: pd.DataFrame, entity_month_wide: pd.DataFrame) -> Dict[str, object]:
    w = entity_month_wide.copy()
    for c in ["K_HPC_kNIS", "K_IPC_kNIS", "K_SIPG_kNIS", "K_HAIFA_TOTAL_kNIS"]:
        w[c] = as_num(w[c]).fillna(0.0)
    diff = w["K_HAIFA_TOTAL_kNIS"] - (w["K_HPC_kNIS"] + w["K_IPC_kNIS"] + w["K_SIPG_kNIS"])
    n_bad = int((diff.abs() > IDENTITY_TOL).sum())
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
    missing_rows = r.loc[r["anchor_observed_flag"].astype(str).str.lower().isin(["false", "0", ""])].copy()
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


def build_qc_summary(anchors, monthly_flows, entity_month_long, entity_month_wide, recon) -> pd.DataFrame:
    checks = [
        qc_annual_flow_identity(anchors, monthly_flows),
        qc_december_anchor_fit(anchors, entity_month_long.loc[entity_month_long["entity"] != "HAIFA_TOTAL"].copy()),
        qc_haifa_sum(entity_month_long, entity_month_wide),
        qc_missing_anchor_years_propagated(recon),
        qc_month_coverage(entity_month_wide),
    ]
    return pd.DataFrame(checks)


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
    qc_summary = build_qc_summary(anchors, monthly_flows, long_df, wide_df, recon)

    # Save final outputs
    write_tsv(long_df, output_dir / "interpolation_02_monthly_entity_series_long.tsv")
    write_tsv(wide_df, output_dir / "interpolation_02_monthly_entity_series_wide.tsv")
    write_tsv(long_df.loc[long_df["entity"] == "HPC"], output_dir / "interpolation_02_monthly_hpc.tsv")
    write_tsv(long_df.loc[long_df["entity"] == "IPC"], output_dir / "interpolation_02_monthly_ipc.tsv")
    write_tsv(long_df.loc[long_df["entity"] == "SIPG"], output_dir / "interpolation_02_monthly_sipg.tsv")
    write_tsv(long_df.loc[long_df["entity"] == "HAIFA_TOTAL"], output_dir / "interpolation_02_monthly_haifa_total.tsv")
    write_tsv(qc_summary, output_dir / "interpolation_02_qc_summary.tsv")

    manifest = {
        "sample_start": SAMPLE_START,
        "sample_end": SAMPLE_END,
        "prepared_dir": str(prepared_dir),
        "output_dir": str(output_dir),
        "rows": {
            "class_capital_rows": int(len(class_capital)),
            "entity_month_rows": int(len(entity_month_all)),
            "qc_checks": int(len(qc_summary)),
        },
        "final_entities": ["HPC", "IPC", "SIPG", "HAIFA_TOTAL"],
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
    print(f"Wrote: {output_dir / 'interpolation_02_build_manifest.json'}")


if __name__ == "__main__":
    main()
