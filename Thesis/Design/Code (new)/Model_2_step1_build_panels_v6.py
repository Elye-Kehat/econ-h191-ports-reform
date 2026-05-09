#!/usr/bin/env python3
"""
Model_2_step1_build_panels_v6.py

Build analysis-ready LP-K/L overlap panels for the current Model 2 pipeline.

This version fixes the v4 failure on the live LP panel.

Main fix
--------
The promoted LP panel stores quarterly rows with `quarter` populated but `month` left
missing. v4 only filled `month` from `quarter` when the entire `month` column was empty,
so quarterly rows kept `month = NaN`. Once terminal / aggregate entities were canonicalized,
multiple quarterly rows in the same calendar year collapsed onto the same merge key
(entity, year, month=NaN), which triggered the duplicate-key error.

v5 fills `month` from `quarter` row-by-row wherever `month` is missing, and then fills
`month_index` row-by-row as well. It also writes a more informative duplicate-key sample
if duplicates still remain for some other reason.

Outputs
-------
Design/Output (new)/Model_2_final/Inputs/
  - model2_terminal_panel.tsv
  - model2_terminal_panel_quarterly.tsv
  - model2_cluster_panel.tsv
  - model2_panel_qa.tsv
  - model2_panel_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

CLUSTER_MAX_YM = 202108


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------

def find_thesis_root(start: Optional[Path] = None) -> Path:
    here = start or Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "Data").exists() and (p / "Design").exists():
            return p
    raise FileNotFoundError("Could not locate thesis root (expected Data/ and Design/).")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def candidate_score(path: Path, kind: str) -> tuple:
    p = str(path).lower()
    name = path.name.lower()
    score = 0

    if kind == "lp":
        if "lp_panel.tsv" in name:
            score += 100
        if "lp_panel" in name:
            score += 60
        if "raw_from_l_v6" in p:
            score += 50
        if "raw_from_l_v5" in p:
            score += 20
        if "model_1a" in p or "model1a" in p:
            score -= 200
    elif kind == "kl":
        if "kl_panel_monthly.tsv" in name:
            score += 100
        if "kl_panel" in name:
            score += 60
        if "common_rule_v6_tonsonly" in p:
            score += 50
        if "model_1b" in p or "model1b" in p:
            score -= 200

    score += int(path.stat().st_mtime)
    return (score,)


def auto_find_panel(root: Path, kind: str) -> Path:
    if kind == "lp":
        candidates = list((root / "Data" / "LP").rglob("*.tsv"))
    elif kind == "kl":
        candidates = list((root / "Data" / "KL").rglob("*.tsv"))
    else:
        raise ValueError(f"Unknown kind: {kind}")

    if not candidates:
        raise FileNotFoundError(f"No TSV candidates found for kind={kind}")

    ranked = sorted(candidates, key=lambda p: candidate_score(p, kind), reverse=True)
    return ranked[0]


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def first_present(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
    return None


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def month_from_quarter(q: object) -> Optional[int]:
    if pd.isna(q):
        return None
    s = str(q).strip().upper()
    mapping = {"Q1": 3, "Q2": 6, "Q3": 9, "Q4": 12, "1": 3, "2": 6, "3": 9, "4": 12}
    return mapping.get(s)


def quarter_from_month(month: object) -> Optional[str]:
    if pd.isna(month):
        return None
    m = int(month)
    if m in (1, 2, 3):
        return "Q1"
    if m in (4, 5, 6):
        return "Q2"
    if m in (7, 8, 9):
        return "Q3"
    if m in (10, 11, 12):
        return "Q4"
    return None


def ensure_month_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "month_index" not in out.columns:
        out["month_index"] = np.nan
    if {"year", "month"}.issubset(out.columns):
        mask = out["year"].notna() & out["month"].notna()
        out.loc[mask, "month_index"] = (
            pd.to_numeric(out.loc[mask, "year"], errors="coerce") * 100
            + pd.to_numeric(out.loc[mask, "month"], errors="coerce")
        )
    return out


def unique_or_null(series: pd.Series) -> object:
    vals = [v for v in series.dropna().unique().tolist()]
    if len(vals) == 1:
        return vals[0]
    return np.nan


def most_common_non_null(series: pd.Series) -> object:
    s = series.dropna()
    if s.empty:
        return np.nan
    return s.mode().iloc[0]


ENTITY_ALIASES: Dict[str, List[str]] = {
    "Haifa--Legacy": [
        "haifa--legacy", "haifa-legacy", "haifa_legacy", "haifa legacy",
        "haifa_legacy_q", "haifa legacy q", "haifa_legacy_kl", "haifa-legacy k/l",
        "haifa--legacy k/l", "haifa legacy terminal", "haifa-legacy terminal",
    ],
    "Haifa--Bayport": [
        "haifa--bayport", "haifa-bayport", "haifa_bayport", "haifa bayport",
        "haifa_sipg", "haifa sipg", "haifa_sipg_q", "haifa sipg q",
        "haifa bayport terminal", "haifa_bayport_kl", "haifa-bayport k/l",
        "haifa--bayport k/l",
    ],
    "Haifa port cluster": [
        "haifa port cluster", "haifa_port_cluster", "haifa_port", "haifa port",
        "haifa_port_q", "haifa", "haifa_port_kl", "haifa port k/l", "haifa aggregate",
    ],
}


def canonical_from_text(text: object) -> Optional[str]:
    if pd.isna(text):
        return None
    s = str(text).strip().lower().replace("s central", "").replace("central", "").strip()
    for canon, aliases in ENTITY_ALIASES.items():
        if s == canon.lower():
            return canon
        if s in aliases:
            return canon
    return None


def canonical_from_row(row: pd.Series) -> Optional[str]:
    text_candidates = [
        row.get("entity"), row.get("series_id"), row.get("target"), row.get("unit"),
        row.get("name"), row.get("label"),
    ]
    for txt in text_candidates:
        hit = canonical_from_text(txt)
        if hit is not None:
            return hit

    port = row.get("port")
    terminal = row.get("terminal")
    if isinstance(port, str):
        p = port.strip().lower()
        t = "" if pd.isna(terminal) else str(terminal).strip().lower()
        if p == "haifa":
            if t in {"legacy", "haifa-legacy", "haifa legacy"}:
                return "Haifa--Legacy"
            if t in {"bayport", "sipg", "haifa-bayport", "haifa bayport", "haifa-sipg", "haifa sipg"}:
                return "Haifa--Bayport"
            if t in {"", "port", "cluster", "aggregate", "haifa-port", "haifa port"}:
                return "Haifa port cluster"
    return None


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def normalize_panel(path: Path, kind: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]

    ren = {}
    c = first_present(df.columns, ["series_id", "entity", "target", "unit", "name", "label"])
    if c and c != "series_id":
        ren[c] = "series_id"
    c = first_present(df.columns, ["port"])
    if c and c != "port":
        ren[c] = "port"
    c = first_present(df.columns, ["terminal"])
    if c and c != "terminal":
        ren[c] = "terminal"
    c = first_present(df.columns, ["year"])
    if c and c != "year":
        ren[c] = "year"
    c = first_present(df.columns, ["month"])
    if c and c != "month":
        ren[c] = "month"
    c = first_present(df.columns, ["quarter"])
    if c and c != "quarter":
        ren[c] = "quarter"
    c = first_present(df.columns, ["month_index", "MonthIndex", "t_index"])
    if c and c != "month_index":
        ren[c] = "month_index"

    if kind == "lp":
        c = first_present(df.columns, ["LP", "lp", "lp_value"])
        if c and c != "LP":
            ren[c] = "LP"
        c = first_present(df.columns, ["log_LP", "ln_LP", "log_lp", "lnlp"])
        if c and c != "log_LP":
            ren[c] = "log_LP"
    elif kind == "kl":
        for src, dst in [
            (["K", "k"], "K"),
            (["L", "l"], "L"),
            (["KL", "k_l", "K_L"], "KL"),
            (["log_KL", "ln_KL", "log_kl", "lnkl"], "log_KL"),
        ]:
            c = first_present(df.columns, src)
            if c and c != dst:
                ren[c] = dst

    df = df.rename(columns=ren)

    if "entity" not in df.columns:
        df["entity"] = df.apply(canonical_from_row, axis=1)

    numeric_cols = [c for c in ["year", "month", "month_index", "LP", "log_LP", "K", "L", "KL", "log_KL", "delta"] if c in df.columns]
    df = coerce_numeric(df, numeric_cols)

    if "month" not in df.columns:
        df["month"] = np.nan
    if "quarter" in df.columns:
        mask_q = df["month"].isna() & df["quarter"].notna()
        df.loc[mask_q, "month"] = df.loc[mask_q, "quarter"].apply(month_from_quarter)

    if "year" not in df.columns:
        df["year"] = np.nan
    if "month_index" in df.columns and df["month_index"].notna().any():
        mask_y = df["year"].isna()
        if mask_y.any():
            df.loc[mask_y, "year"] = (pd.to_numeric(df.loc[mask_y, "month_index"], errors="coerce") // 100)
        mask_m = df["month"].isna()
        if mask_m.any():
            df.loc[mask_m, "month"] = (pd.to_numeric(df.loc[mask_m, "month_index"], errors="coerce") % 100)

    df = ensure_month_index(df)

    if kind == "lp":
        if "log_LP" not in df.columns and "LP" in df.columns:
            vals = pd.to_numeric(df["LP"], errors="coerce")
            df["log_LP"] = np.where(vals > 0, np.log(vals), np.nan)
    elif kind == "kl":
        if "log_KL" not in df.columns and "KL" in df.columns:
            vals = pd.to_numeric(df["KL"], errors="coerce")
            df["log_KL"] = np.where(vals > 0, np.log(vals), np.nan)

    df["entity"] = df.apply(canonical_from_row, axis=1)

    keep = [c for c in [
        "entity", "series_id", "port", "terminal", "year", "month", "quarter", "month_index",
        "freq", "delta", "dep_scenario", "LP", "log_LP", "K", "L", "KL", "log_KL",
    ] if c in df.columns]
    out = df[keep].copy()
    out["source_kind"] = kind
    out["source_file"] = str(path)
    return out


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def duplicate_sample(df: pd.DataFrame, keys: List[str], n: int = 10) -> pd.DataFrame:
    if df.empty:
        return df.head(0)
    mask = df.duplicated(keys, keep=False)
    extra_priority = [c for c in ["series_id", "port", "terminal", "quarter", "freq", "LP", "KL", "log_LP", "log_KL", "source_file"] if c in df.columns and c not in keys]
    extra_rest = [c for c in df.columns if c not in keys and c not in extra_priority]
    cols = keys + extra_priority + extra_rest
    return df.loc[mask, cols].sort_values(keys).head(n)


def summarize_panel(df: pd.DataFrame, name: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if df.empty:
        return [{"panel": name, "entity": "<EMPTY>", "rows": 0}]

    for entity, sub in df.groupby("entity", dropna=False):
        rows.append({
            "panel": name,
            "entity": entity if pd.notna(entity) else "<NA>",
            "rows": int(len(sub)),
            "year_min": float(sub["year"].min()) if "year" in sub.columns and sub["year"].notna().any() else np.nan,
            "year_max": float(sub["year"].max()) if "year" in sub.columns and sub["year"].notna().any() else np.nan,
            "month_index_min": float(sub["month_index"].min()) if "month_index" in sub.columns and sub["month_index"].notna().any() else np.nan,
            "month_index_max": float(sub["month_index"].max()) if "month_index" in sub.columns and sub["month_index"].notna().any() else np.nan,
            "missing_log_LP": int(sub["log_LP"].isna().sum()) if "log_LP" in sub.columns else np.nan,
            "missing_log_KL": int(sub["log_KL"].isna().sum()) if "log_KL" in sub.columns else np.nan,
        })
    return rows


def collapse_terminal_to_quarter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    d = df.copy()
    d["quarter"] = d["month"].apply(quarter_from_month)
    qnum = d["quarter"].map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4})
    d["quarter_index"] = d["year"].astype(int) * 10 + qnum.astype(int)

    rows = []
    for keys, sub in d.groupby(["entity", "year", "quarter", "quarter_index"], dropna=False):
        row = dict(zip(["entity", "year", "quarter", "quarter_index"], keys))
        row["port"] = most_common_non_null(sub.get("port", pd.Series(dtype=object)))
        row["terminal"] = most_common_non_null(sub.get("terminal", pd.Series(dtype=object)))
        row["LP"] = unique_or_null(sub["LP"]) if "LP" in sub.columns else np.nan
        if pd.isna(row["LP"]) and "LP" in sub.columns:
            row["LP"] = most_common_non_null(sub["LP"])
        row["log_LP"] = np.log(row["LP"]) if pd.notna(row["LP"]) and row["LP"] > 0 else np.nan
        for col in ["K", "L", "KL", "log_KL"]:
            row[col] = float(sub[col].mean()) if col in sub.columns and sub[col].notna().any() else np.nan
        row["n_months_in_quarter"] = int(len(sub))
        row["month_index_min"] = int(sub["month_index"].min())
        row["month_index_max"] = int(sub["month_index"].max())
        row["lp_freq"] = "Q"
        row["kl_freq"] = "Qmean_from_M"
        row["source_lp"] = most_common_non_null(sub.get("source_lp", pd.Series(dtype=object)))
        row["source_kl"] = most_common_non_null(sub.get("source_kl", pd.Series(dtype=object)))
        row["lp_source_file"] = most_common_non_null(sub.get("lp_source_file", pd.Series(dtype=object)))
        row["kl_source_file"] = most_common_non_null(sub.get("kl_source_file", pd.Series(dtype=object)))
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["entity", "year", "quarter_index"]).reset_index(drop=True)
    out["t_index"] = out.groupby("entity").cumcount() + 1
    return out


# ---------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------

def build_panels(lp_path: Path, kl_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    lp = normalize_panel(lp_path, kind="lp")
    kl = normalize_panel(kl_path, kind="kl")

    keep_entities = ["Haifa--Legacy", "Haifa--Bayport", "Haifa port cluster"]
    lp = lp[lp["entity"].isin(keep_entities)].copy()
    kl = kl[kl["entity"].isin(keep_entities)].copy()

    lp = lp[~((lp["entity"] == "Haifa port cluster") & ((lp["year"] * 100 + lp["month"]) > CLUSTER_MAX_YM))].copy()
    kl = kl[~((kl["entity"] == "Haifa port cluster") & ((kl["year"] * 100 + kl["month"]) > CLUSTER_MAX_YM))].copy()

    merge_keys = ["entity", "year", "month"]

    lp_dup = duplicate_sample(lp, merge_keys)
    kl_dup = duplicate_sample(kl, merge_keys)
    if not lp_dup.empty:
        sample_path = out_dir / "model2_lp_duplicate_key_sample.tsv"
        lp_dup.to_csv(sample_path, sep="\t", index=False)
        raise ValueError(f"LP panel has duplicate merge keys. See {sample_path}")
    if not kl_dup.empty:
        sample_path = out_dir / "model2_kl_duplicate_key_sample.tsv"
        kl_dup.to_csv(sample_path, sep="\t", index=False)
        raise ValueError(f"K/L panel has duplicate merge keys. See {sample_path}")

    lp_cols = [c for c in ["entity", "series_id", "port", "terminal", "year", "month", "month_index", "freq", "delta", "dep_scenario", "LP", "log_LP", "source_file"] if c in lp.columns]
    kl_cols = [c for c in ["entity", "series_id", "port", "terminal", "year", "month", "month_index", "freq", "delta", "dep_scenario", "K", "L", "KL", "log_KL", "source_file"] if c in kl.columns]

    lp_small = lp[lp_cols].copy().rename(columns={
        "series_id": "lp_series_id", "port": "lp_port", "terminal": "lp_terminal",
        "freq": "lp_freq", "delta": "lp_delta", "dep_scenario": "lp_dep_scenario",
        "source_file": "lp_source_file",
    })
    kl_small = kl[kl_cols].copy().rename(columns={
        "series_id": "kl_series_id", "port": "kl_port", "terminal": "kl_terminal",
        "freq": "kl_freq", "delta": "kl_delta", "dep_scenario": "kl_dep_scenario",
        "source_file": "kl_source_file",
    })

    merge_on = merge_keys
    merged = lp_small.merge(kl_small, on=merge_on, how="inner", validate="one_to_one")

    merged["port"] = merged.get("lp_port").combine_first(merged.get("kl_port")) if "lp_port" in merged.columns and "kl_port" in merged.columns else merged.get("lp_port", merged.get("kl_port"))
    merged["terminal"] = merged.get("lp_terminal").combine_first(merged.get("kl_terminal")) if "lp_terminal" in merged.columns and "kl_terminal" in merged.columns else merged.get("lp_terminal", merged.get("kl_terminal"))
    merged["month_index"] = (
        pd.to_numeric(merged["year"], errors="coerce") * 100
        + pd.to_numeric(merged["month"], errors="coerce")
    )
    merged = merged.sort_values(["entity", "year", "month_index"]).reset_index(drop=True)
    merged["t_index"] = merged.groupby("entity").cumcount() + 1
    merged["source_lp"] = merged.get("lp_series_id", pd.Series(index=merged.index, dtype="object"))
    merged["source_kl"] = merged.get("kl_series_id", pd.Series(index=merged.index, dtype="object"))

    terminal_panel = merged[merged["entity"].isin(["Haifa--Legacy", "Haifa--Bayport"])].copy()
    cluster_panel = merged[merged["entity"] == "Haifa port cluster"].copy()
    terminal_quarterly = collapse_terminal_to_quarter(terminal_panel)

    keep_common = [
        "entity", "port", "terminal", "year", "month", "month_index", "t_index",
        "LP", "log_LP", "K", "L", "KL", "log_KL",
        "lp_freq", "kl_freq", "lp_delta", "kl_delta", "lp_dep_scenario", "kl_dep_scenario",
        "source_lp", "source_kl", "lp_source_file", "kl_source_file",
    ]
    terminal_out = terminal_panel[[c for c in keep_common if c in terminal_panel.columns]].copy()
    cluster_out = cluster_panel[[c for c in keep_common if c in cluster_panel.columns]].copy()

    terminal_q_cols = [
        "entity", "port", "terminal", "year", "quarter", "quarter_index", "t_index",
        "LP", "log_LP", "K", "L", "KL", "log_KL", "n_months_in_quarter",
        "month_index_min", "month_index_max", "lp_freq", "kl_freq",
        "source_lp", "source_kl", "lp_source_file", "kl_source_file",
    ]
    terminal_q_out = terminal_quarterly[[c for c in terminal_q_cols if c in terminal_quarterly.columns]].copy()

    terminal_path = out_dir / "model2_terminal_panel.tsv"
    terminal_q_path = out_dir / "model2_terminal_panel_quarterly.tsv"
    cluster_path = out_dir / "model2_cluster_panel.tsv"
    qa_path = out_dir / "model2_panel_qa.tsv"
    manifest_path = out_dir / "model2_panel_manifest.json"

    terminal_out.to_csv(terminal_path, sep="\t", index=False)
    terminal_q_out.to_csv(terminal_q_path, sep="\t", index=False)
    cluster_out.to_csv(cluster_path, sep="\t", index=False)

    qa_rows: List[Dict[str, object]] = []
    qa_rows.extend(summarize_panel(terminal_out, "terminal_monthly"))
    term_q_for_qa = terminal_q_out.rename(columns={"quarter_index": "month_index"}).copy()
    qa_rows.extend(summarize_panel(term_q_for_qa, "terminal_quarterly"))
    qa_rows.extend(summarize_panel(cluster_out, "cluster_monthly"))
    pd.DataFrame(qa_rows).to_csv(qa_path, sep="\t", index=False)

    manifest = {
        "lp_input": str(lp_path),
        "kl_input": str(kl_path),
        "lp_sha256": sha256_file(lp_path),
        "kl_sha256": sha256_file(kl_path),
        "cluster_max_ym_cutoff": CLUSTER_MAX_YM,
        "terminal_rows_monthly": int(len(terminal_out)),
        "terminal_rows_quarterly": int(len(terminal_q_out)),
        "cluster_rows": int(len(cluster_out)),
        "terminal_entities": sorted([str(x) for x in terminal_out["entity"].dropna().unique().tolist()]),
        "cluster_entities": sorted([str(x) for x in cluster_out["entity"].dropna().unique().tolist()]),
        "merge_keys": merge_keys,
        "note": "Quarter-collapsed terminal panel is the preferred elasticity input; month_index is standardized to YYYYMM and merge keys use only (entity, year, month).",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== Model_2_step1_build_panels_v6.py: done ===")
    print(f"LP input           : {lp_path}")
    print(f"K/L input          : {kl_path}")
    print(f"Terminal out       : {terminal_path}")
    print(f"Terminal quarterly : {terminal_q_path}")
    print(f"Cluster out        : {cluster_path}")
    print(f"QA out             : {qa_path}")
    print(f"Manifest           : {manifest_path}")
    print(f"Terminal rows (M)  : {len(terminal_out)}")
    print(f"Terminal rows (Q)  : {len(terminal_q_out)}")
    print(f"Cluster rows       : {len(cluster_out)}")


def main() -> None:
    thesis_root = find_thesis_root()

    parser = argparse.ArgumentParser(description="Build Model 2 LP-K/L overlap panels.")
    parser.add_argument("--lp", type=Path, default=None, help="Path to the final LP panel used by Model 1A.")
    parser.add_argument("--kl", type=Path, default=None, help="Path to the final K/L panel used by Model 1B.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=thesis_root / "Design" / "Output (new)" / "Model_2_final" / "Inputs",
        help="Output directory for Model 2 input panels.",
    )
    args = parser.parse_args()

    lp_path = args.lp if args.lp is not None else auto_find_panel(thesis_root, "lp")
    kl_path = args.kl if args.kl is not None else auto_find_panel(thesis_root, "kl")

    if not lp_path.exists():
        raise FileNotFoundError(f"LP input not found: {lp_path}")
    if not kl_path.exists():
        raise FileNotFoundError(f"K/L input not found: {kl_path}")

    build_panels(lp_path, kl_path, args.outdir)


if __name__ == "__main__":
    main()
