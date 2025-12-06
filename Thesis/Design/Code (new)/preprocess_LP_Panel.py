#!/usr/bin/env python
"""
preprocess_LP_Panel.py

Purpose
-------
Take the mixed-frequency LP panel (monthly ports, quarterly terminals)
stored in Data/LP/LP_Panel.tsv and produce a purely monthly panel
Data/LP/LP_Panel_monthly.tsv.

Logic
-----
- Keep all rows that are already monthly (freq == 'M') as they are.
- For quarterly rows (freq == 'Q'):
    * Use 'quarter' and 'year' to determine which months belong to that
      quarter (Q1 = Jan–Mar, ..., Q4 = Oct–Dec).
    * For each month in the quarter, create a new row:
        - copy all columns from the quarterly row,
        - set freq = 'M',
        - set 'month' to the calendar month (1–12),
        - set 'month_index' using a global (year,month)->month_index map.
          If the month is beyond the last existing month in the map, we
          extend the map linearly (month_index + 1 each month).
    * This makes LP and all other variables step functions within the quarter.
- Drop the original quarterly rows.

The output panel is fully monthly and ready for NYT/event-study code.
"""

from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
THESIS_ROOT = THIS_FILE.parents[2]  # .../Thesis/Design/Code (new)/preprocess_LP_Panel.py

INPUT_PATH = THESIS_ROOT / "Data" / "LP" / "LP_Panel.tsv"
OUTPUT_PATH = THESIS_ROOT / "Data" / "LP" / "LP_Panel_monthly.tsv"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def build_base_mapping(df_monthly: pd.DataFrame) -> Dict[Tuple[int, int], int]:
    """
    Build the initial mapping (year, month) -> month_index from rows that
    are already monthly (freq == 'M') and have year/month/month_index defined.

    We require a unique month_index per (year, month).
    """
    mapping: Dict[Tuple[int, int], int] = {}

    mask = (
        df_monthly["year"].notna()
        & df_monthly["month"].notna()
        & df_monthly["month_index"].notna()
    )
    grouped = df_monthly.loc[mask].groupby(["year", "month"])["month_index"].unique()

    for (y, m), idxs in grouped.items():
        idxs = [i for i in idxs if pd.notna(i)]
        if len(idxs) == 0:
            continue
        if len(idxs) > 1:
            raise ValueError(
                f"(year={y}, month={m}) has multiple month_index values: {idxs}. "
                "LP_Panel should use a single global month_index per month."
            )
        mapping[(int(y), int(m))] = int(idxs[0])

    if not mapping:
        raise ValueError(
            "No (year, month, month_index) combinations found. "
            "Check that the monthly series in LP_Panel.tsv have month_index populated."
        )

    return mapping


def quarter_to_months(quarter_label: str) -> List[int]:
    """
    Convert a quarter label like 'Q1', 'Q2', 'Q3', 'Q4' into the
    list of calendar months in that quarter.
    """
    if not isinstance(quarter_label, str) or not quarter_label.startswith("Q"):
        raise ValueError(f"Quarter label '{quarter_label}' is not of form 'Q1'...'Q4'.")

    qnum = quarter_label[1:]
    if qnum == "1":
        return [1, 2, 3]
    elif qnum == "2":
        return [4, 5, 6]
    elif qnum == "3":
        return [7, 8, 9]
    elif qnum == "4":
        return [10, 11, 12]
    else:
        raise ValueError(f"Unknown quarter label '{quarter_label}'.")


def get_or_create_month_index(mapping: Dict[Tuple[int, int], int],
                              year: int,
                              month: int) -> int:
    """
    Return month_index for (year, month). If it doesn't exist yet and the
    requested month is after the last known month, extend the mapping
    linearly (month_index + 1 per month).

    We assume we process months in non-decreasing chronological order,
    so we never need to "fill holes" backwards.
    """
    key = (year, month)
    if key in mapping:
        return mapping[key]

    # Extend beyond the last known month
    last_year, last_month = max(mapping.keys())
    last_idx = mapping[(last_year, last_month)]

    last_total = last_year * 12 + last_month
    new_total = year * 12 + month

    if new_total <= last_total:
        raise ValueError(
            f"Requested (year={year}, month={month}) is not after "
            f"last known (year={last_year}, month={last_month}). "
            "This suggests the quarterly data are not processed in chronological order."
        )

    # Number of months ahead
    diff_months = new_total - last_total
    new_idx = last_idx + diff_months

    mapping[key] = new_idx
    return new_idx


# ----------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------

def main():
    print(f"Reading LP panel from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, sep="\t")

    # Ensure year/month/month_index are numeric (nullable Int64 to allow NA)
    for col in ["year", "month", "month_index"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Split into monthly and quarterly parts
    monthly_mask = df["freq"] == "M"
    quarterly_mask = df["freq"] == "Q"

    df_monthly = df[monthly_mask].copy()
    df_quarterly = df[quarterly_mask].copy()

    print(f"Monthly rows:   {len(df_monthly)}")
    print(f"Quarterly rows: {len(df_quarterly)}")

    if df_quarterly.empty:
        print("No quarterly rows found; copying LP_Panel.tsv to LP_Panel_monthly.tsv.")
        df.to_csv(OUTPUT_PATH, sep="\t", index=False)
        print(f"Done. Wrote {len(df)} rows to {OUTPUT_PATH}")
        return

    # Build initial (year, month) -> month_index mapping
    ym_to_idx = build_base_mapping(df_monthly)
    print(f"Built base (year, month) -> month_index mapping for {len(ym_to_idx)} months.")
    print(f"Last known month in base mapping: {max(ym_to_idx.keys())}")

    # Expand quarterly rows to monthly
    expanded_rows = []

    # Sort quarterly rows by year, quarter, series_id to ensure we move forward in time
    dfq_sorted = df_quarterly.sort_values(["year", "quarter", "series_id"]).copy()

    for _, row in dfq_sorted.iterrows():
        year = int(row["year"])
        q_label = row["quarter"]
        months_in_q = quarter_to_months(q_label)

        for m in months_in_q:
            idx = get_or_create_month_index(ym_to_idx, year, m)

            new_row = row.copy()
            new_row["freq"] = "M"
            new_row["month"] = m
            new_row["month_index"] = idx

            expanded_rows.append(new_row)

    df_expanded = pd.DataFrame(expanded_rows)
    print(f"Created {len(df_expanded)} monthly rows from {len(df_quarterly)} quarterly rows.")
    print(f"Final (year, month) -> month_index mapping now has {len(ym_to_idx)} months.")
    print(f"Last month in mapping: {max(ym_to_idx.keys())}")

    # Combine original monthly and expanded terminal rows
    df_out = pd.concat([df_monthly, df_expanded], ignore_index=True)

    # Optional: sort for readability
    sort_cols = [c for c in ["series_id", "year", "month_index"] if c in df_out.columns]
    df_out = df_out.sort_values(sort_cols).reset_index(drop=True)

    # Sanity check: all rows are now monthly
    assert set(df_out["freq"].unique()) == {"M"}, "Output panel should be all monthly (freq == 'M')."

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, sep="\t", index=False)

    print(f"Done. Wrote {len(df_out)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
