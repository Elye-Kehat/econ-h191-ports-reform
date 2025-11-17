import pandas as pd
import numpy as np
from pathlib import Path
import re

# ---------- Resolve project paths ----------
HERE = Path(__file__).resolve()
# Econometrics -> Code -> Design -> Thesis
THESIS_ROOT = HERE.parents[3]

# INPUTS
lp_path = THESIS_ROOT / "Data" / "LP" / "LP_Panel.tsv"
q_path  = THESIS_ROOT / "Design" / "Output Data" / "01_panel_port_quarter_full.csv"  # <- was panel_port_quarter_full.csv


# OUTPUT (inside Thesis / Design / Output Data)
out_dir  = THESIS_ROOT / "Design" / "Output Data"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "02_LP_Panel_quarterized.tsv"

print("[paths]")
print("  LP_Panel:", lp_path)
print("  quarterized CSV:", q_path)
print("  OUT (TSV):", out_path)

# --- 1) Load
lp = pd.read_csv(lp_path, sep="\t")
q_raw = pd.read_csv(q_path)

# --- 2–3) Identify & drop monthly rows in LP
# Prefer freq/frequency if present; fall back to 'month' existence.
def _find_case_insensitive(df, names):
    low = {c.lower(): c for c in df.columns}
    for want in names:
        if want in low:
            return low[want]
    return None

freq_col = _find_case_insensitive(lp, {"frequency","freq"})
if freq_col:
    monthly_mask = lp[freq_col].astype(str).str.lower().str.startswith("m")
else:
    # fallback: presence of a 'month' column
    month_col = _find_case_insensitive(lp, {"month"})
    monthly_mask = lp[month_col].notna() if month_col else pd.Series(False, index=lp.index)

lp_non_monthly = lp.loc[~monthly_mask, :].copy()

# --- Helper: find a column by case-insensitive exact name
def find_col(df, want):
    for c in df.columns:
        if c.strip().lower() == want:
            return c
    return None

# --- 4) Build the base subset (LP) with the exact 5 columns
colmap_lp = {find_col(lp_non_monthly, w): w
             for w in ["port","terminal","year","quarter","lp"]
             if find_col(lp_non_monthly, w)}
lp_small = lp_non_monthly.rename(columns=colmap_lp)
for c in ["port","terminal","year","quarter","lp"]:
    if c not in lp_small.columns:
        lp_small[c] = np.nan
# normalize types a bit
lp_small["port"] = lp_small["port"].astype(str).str.strip()
if lp_small["terminal"].isna().any():
    lp_small["terminal"] = lp_small["terminal"].fillna("")
lp_small = lp_small[["port","terminal","year","quarter","lp"]].copy()

# --- Parse quarterized CSV: 'qtr' like 2018Q1 -> (year, quarter)
assert {"port","qtr","LP"}.issubset(set(q_raw.columns)), \
    "Expected columns 'port','qtr','LP' in panel_port_quarter_full.csv"

yx = q_raw["qtr"].astype(str).str.extract(r"^(\d{4})Q([1-4])$", expand=True)
q_parsed = q_raw.copy()
q_parsed["year"] = yx[0]
q_parsed["quarter"] = yx[1]
q_parsed["lp"] = pd.to_numeric(q_parsed["LP"], errors="coerce")

q_small = q_parsed[["port","year","quarter","lp"]].copy()
# port-level quarterized rows: terminal blank by design
q_small["terminal"] = ""
q_small = q_small[["port","terminal","year","quarter","lp"]]

# --- 5) Anti-join on (port, terminal, year, quarter)
def make_key(df):
    return (df["port"].astype(str).str.strip().fillna("")
            + "||" + df["terminal"].astype(str).str.strip().fillna("")
            + "||" + df["year"].astype(str).str.strip().fillna("")
            + "||" + df["quarter"].astype(str).str.strip().fillna(""))

lp_small["__key__"] = make_key(lp_small)
q_small["__key__"] = make_key(q_small)

base = lp_small[~lp_small["__key__"].isin(q_small["__key__"])].copy()

# --- 6–7) Append, dedupe, sort, save
merged = pd.concat([base.drop(columns="__key__"),
                    q_small.drop(columns="__key__")],
                   ignore_index=True)
merged = merged.drop_duplicates(subset=["port","terminal","year","quarter"], keep="last")
merged = merged.sort_values(["port","terminal","year","quarter"], kind="mergesort")

merged.to_csv(out_path, sep="\t", index=False)

# Optional: quick diagnostics
print("[done]", {
    "lp_rows_total": len(lp),
    "lp_rows_non_monthly_kept": len(lp_non_monthly),
    "quarterized_rows": len(q_small),
    "merged_rows": len(merged),
    "output_path": str(out_path)
})
