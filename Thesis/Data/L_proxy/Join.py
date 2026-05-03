#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge delta into master and write L_Proxy.tsv (TSV)

Usage:
  python Data/L_proxy/merge_L_delta_into_master.py \
    --master Data/L_proxy/labor_hours_monthly_terminal.tsv \
    --delta  Data/L_proxy/labor_hours_missing_delta.tsv \
    --out    Data/L_proxy/L_Proxy.tsv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# --- Helpers ---
DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"

def norm_dashes(x: str):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return x
    s = str(x)
    for d in DASHES:
        s = s.replace(d, "-")
    return s

def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\t|,", engine="python")

def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)

def to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")

def month_to_quarter(m):
    if pd.isna(m): return np.nan
    m = int(m)
    return {1:"Q1",2:"Q1",3:"Q1",4:"Q2",5:"Q2",6:"Q2",7:"Q3",8:"Q3",9:"Q3",10:"Q4",11:"Q4",12:"Q4"}[m]

def main(master_path: Path, delta_path: Path, out_path: Path) -> None:
    # --- Load ---
    master = read_table(master_path)
    delta  = read_table(delta_path)

    # --- Canonicalize labels & dtypes ---
    for df in (master, delta):
        if "port" in df.columns:     df["port"]     = df["port"].map(norm_dashes)
        if "terminal" in df.columns: df["terminal"] = df["terminal"].map(norm_dashes)

    for c in ["year","month"]:
        if c in master.columns: master[c] = to_int(master[c])
        if c in delta.columns:  delta[c]  = to_int(delta[c])

    # Ensure quarter exists in master
    if "quarter" not in master.columns and "month" in master.columns:
        master["quarter"] = master["month"].map(month_to_quarter)

    # --- Convert delta to master schema ---
    # We only need the “_new” columns and rename them to master names.
    need_from_delta = [
        "port","terminal","year","month","quarter",
        "TEU_port_m","share_i_p_q","TEU_i_m",
        "Pi_teu_per_hour_i_y_new","H_annual_i_y_new","w_i_m_in_year","L_hours_i_m_new"
    ]
    # Keep only columns that exist (in case delta has extra debug cols)
    need_from_delta = [c for c in need_from_delta if c in delta.columns]
    dstd = delta[need_from_delta].copy()

    rename_map = {
        "Pi_teu_per_hour_i_y_new": "Pi_teu_per_hour_i_y",
        "H_annual_i_y_new": "H_annual_i_y",
        "L_hours_i_m_new": "L_hours_i_m",
    }
    dstd = dstd.rename(columns=rename_map)

    # Ensure delta has the same dtype for keys
    for c in ["year","month"]:
        if c in dstd.columns: dstd[c] = to_int(dstd[c])

    # --- Compute update/insert counts before merging (for reporting only) ---
    def key_tuples(df: pd.DataFrame):
        return set(zip(df.get("port"), df.get("terminal"), df.get("year"), df.get("month")))
    master_keys = key_tuples(master)
    delta_keys  = key_tuples(dstd)
    n_update = len(master_keys & delta_keys)
    n_insert = len(delta_keys - master_keys)

    # --- Outer merge on keys; prefer delta values when present ---
    key_cols = ["port","terminal","year","month"]
    # Keep only columns present to avoid KeyError if a required column is missing
    key_cols = [c for c in key_cols if c in master.columns and c in dstd.columns]

    merged = master.merge(dstd, on=key_cols, how="outer", suffixes=("", "_d"))

    # Columns we will update/insert from delta (if delta provided them)
    candidate_cols = ["quarter","TEU_port_m","share_i_p_q","TEU_i_m",
                      "Pi_teu_per_hour_i_y","H_annual_i_y","w_i_m_in_year","L_hours_i_m"]

    for c in candidate_cols:
        c_d = f"{c}_d"
        if c in merged.columns and c_d in merged.columns:
            # Prefer delta value when available; otherwise keep master
            merged[c] = merged[c_d].combine_first(merged[c])
            merged = merged.drop(columns=[c_d])
        elif c_d in merged.columns and c not in merged.columns:
            # Column missing in master; create it from delta
            merged[c] = merged[c_d]
            merged = merged.drop(columns=[c_d])
        # else: nothing to do (neither side had the column)

    # If master had columns not in delta, they are already preserved by merge.

    # (Optional) sort and tidy
    sort_cols = [c for c in ["port","terminal","year","month"] if c in merged.columns]
    merged = merged.sort_values(sort_cols).reset_index(drop=True)

    # Write
    write_tsv(merged, out_path)

    print(f"✔ Wrote merged table: {out_path}")
    print(f"   Updated existing keys: {n_update}")
    print(f"   Inserted new keys:     {n_insert}")
    print(f"   Final rows:            {len(merged)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge missing-only delta into master terminal table (TSV).")
    ap.add_argument("--master", type=Path, default=Path("Data/L_proxy/labor_hours_monthly_terminal.tsv"))
    ap.add_argument("--delta",  type=Path, default=Path("Data/L_proxy/labor_hours_missing_delta.tsv"))
    ap.add_argument("--out",    type=Path, default=Path("Data/L_proxy/L_Proxy.tsv"))
    args = ap.parse_args()
    main(args.master, args.delta, args.out)
