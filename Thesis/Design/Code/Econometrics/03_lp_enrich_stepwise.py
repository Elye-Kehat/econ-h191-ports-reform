
#!/usr/bin/env python3
"""
Stepwise enrichment for LP_Panel_quarterized.tsv — robust parsing (v2)

Improvements:
- Cleans 'year' (handles '2024', '2024.0', 'Year:2024')
- Cleans 'quarter' (handles '1', '1.0', 'Q1', 'q1', ' Q2 ')
- Writes a 'cleaning_log.tsv' with counts of fixes by rule
- Writes a 'bad_year_or_quarter_rows.tsv' only if problems remain

Steps (toggle):
  1) Parse quarter string `qtr = YYYYQ#` (ON by default)
  2) Add outcome `Y = ln(lp)`
  3) Add global time index `t_index` over sorted unique qtrs
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re

# ------------------------------
# User toggles
# ------------------------------
RUN_STEP_1 = True   # qtr parsing
RUN_STEP_2 = True  # add Y = ln(lp)
RUN_STEP_3 = True  # add t_index

# ------------------------------
# Locate input
# ------------------------------
PREFERRED = Path("/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis/Design/Output Data/02_LP_Panel_quarterized.tsv")
FALLBACK  = Path("Design/Output Data/02_LP_Panel_quarterized.tsv")


if PREFERRED.exists():
    inp = PREFERRED
    out_dir = PREFERRED.parent
elif FALLBACK.exists():
    inp = FALLBACK
    out_dir = FALLBACK.parent
else:
    sys.exit("Could not find LP_Panel_quarterized.tsv in either path.")

print(f"[input] {inp.resolve()}")
df = pd.read_csv(inp, sep="\t")

required_cols = {"port","terminal","year","quarter","lp"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# ------------------------------
# Cleaning helpers
# ------------------------------
def clean_year(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    # Extract 4 consecutive digits
    extracted = s.str.extract(r"(\d{4})", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")

def clean_quarter(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    # Normalize common forms: 'Q1', '1', '1.0', ' Q2 ', etc.
    # Extract a single digit 1-4, possibly preceded by 'Q'
    extracted = s.str.extract(r"Q?\s*([1-4])", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")

# ------------------------------
# Apply cleaning BEFORE validation
# ------------------------------
clean_log = []

# Year cleaning
year_before_na = df["year"].isna().sum()
df["year_raw"] = df["year"]
df["year"] = clean_year(df["year"])
year_after_na = df["year"].isna().sum()
clean_log.append({"field":"year","n_fixed": int(year_before_na - year_after_na)})

# Quarter cleaning
q_before_na = df["quarter"].isna().sum()
df["quarter_raw"] = df["quarter"]
df["quarter"] = clean_quarter(df["quarter"])
q_after_na = df["quarter"].isna().sum()
clean_log.append({"field":"quarter","n_fixed": int(q_before_na - q_after_na)})

# Validate
bad_mask = df["year"].isna() | df["quarter"].isna() | ~df["quarter"].isin([1,2,3,4])
if bad_mask.any():
    bad = df.loc[bad_mask, ["port","terminal","year_raw","quarter_raw","year","quarter"]]
    bad.to_csv(out_dir / "03_bad_year_or_quarter_rows.tsv", sep="\t", index=False)
    print("[warn] Still found invalid/missing year or quarter after cleaning. See 'bad_year_or_quarter_rows.tsv'.")
else:
    # Drop raw columns after successful cleaning
    df = df.drop(columns=["year_raw","quarter_raw"], errors="ignore")

# Write cleaning log
pd.DataFrame(clean_log).to_csv(out_dir / "03_cleaning_log.tsv", sep="\t", index=False)
# ----------------------------------
# STEP 1: qtr parsing (YYYYQ#)
# ----------------------------------
if RUN_STEP_1:
    # If invalid remain, we still compute qtr for valid rows and leave NaN for bad ones
    df["qtr"] = np.where(
        df["year"].notna() & df["quarter"].isin([1,2,3,4]),
        df["year"].astype("Int64").astype(str) + "Q" + df["quarter"].astype("Int64").astype(str),
        pd.NA
    )
    step1 = out_dir / "03_LP_Panel_quarterized.step1_qtr.tsv"
    df.to_csv(step1, sep="\t", index=False)
    print(f"[step1:qtr] wrote {step1}")

# ----------------------------------
# STEP 2: add Y = ln(lp)
# ----------------------------------
if RUN_STEP_2:
    nonpos = (df["lp"] <= 0)
    if nonpos.any():
        df.loc[nonpos, "lp"] = np.nan
        print(f"[warn] Replaced {int(nonpos.sum())} non-positive lp values with NaN before log.")
    df["Y"] = np.log(df["lp"])
    step2 = out_dir / "03_LP_Panel_quarterized.step2_qtr_Y.tsv"
    df.to_csv(step2, sep="\t", index=False)
    print(f"[step2:Y] wrote {step2}")

# ----------------------------------
# STEP 3: add t_index (global quarter index)
# ----------------------------------
if RUN_STEP_3:
    valid = df[df["qtr"].notna()].copy()
    uq = (
        valid[["year","quarter"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["year","quarter"])
        .assign(qtr=lambda x: x["year"].astype(int).astype(str) + "Q" + x["quarter"].astype(int).astype(str))
    )
    q_map = {q: i for i, q in enumerate(uq["qtr"].tolist())}
    df["t_index"] = df["qtr"].map(q_map).astype("Int64")
    step3 = out_dir / "03_LP_Panel_quarterized.step3_qtr_Y_tindex.tsv"
    df.to_csv(step3, sep="\t", index=False)
    print(f"[step3:t_index] wrote {step3}")

print("[done]")
