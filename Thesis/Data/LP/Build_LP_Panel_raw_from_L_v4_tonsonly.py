#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build_LP_Panel_raw_from_L_v4_tonsonly.py
=======================================

Purpose
-------
Fix the monthly port LP branch in v3.

Why v4 exists
-------------
The v3 tons-only LP build correctly produced the quarterly terminal LP objects, but the
monthly port LP objects could become entirely missing because the monthly labor merge
required an exact match on `month_index` across:
    - Stage 2 TEU monthly tables, and
    - the tons-only L_Proxy.

Those two upstream branches do not use the same month_index convention:
    * Stage 2 TEU tables use a running month index like year*12 + month
    * the tons-only L_Proxy uses YYYYMM

As a result, the v3 monthly-port merge could silently fail even when the same
(port, year, month) rows existed in both inputs. Once that happened:
    - L_hours_port_m became NaN
    - LP_raw became NaN
    - pre-reform port LP rows were dropped by Model 1A
    - the competition incumbent splice lost its pre-period support

Core fix in v4
--------------
For the monthly port LP branch only:
    1) aggregate labor by (port, year, month)
    2) merge TEU, labor, and tons by (port, year, month) rather than month_index
    3) keep the TEU-side month_index as the output month_index
    4) add hard QA checks that fail if a port's monthly LP is entirely missing

What does NOT change
--------------------
The LP definition remains the current active raw-from-L definition:

    Monthly port LP:
        LP_{p,m} = TEU_{p,m} / L_{p,m}

    Quarterly terminal LP:
        LP_{i,q} = TEU_{i,q} / L_{i,q}
        where L_{i,q} = sum_{m in q} L_{i,m}

Primary outputs
---------------
By default, outputs are written to:
    Data/LP/raw_from_l_v4_tonsonly/

and include:
    LP_Haifa_port_month.tsv
    LP_Ashdod_port_month.tsv
    LP_Haifa_Legacy_quarter.tsv
    LP_Haifa_SIPG_quarter.tsv
    LP_Ashdod_Legacy_quarter.tsv
    LP_Ashdod_HCT_quarter.tsv
    LP_Panel.tsv
    LP_raw_qa.tsv
    _meta_lp_raw_v4_tonsonly.json

Optional convenience
--------------------
If --also-write-canonical is passed, the script also writes:
    Data/LP/LP_Panel.tsv

so Model 1A v7 can read the new LP panel through the canonical path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

TAB = "\t"


def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=TAB, engine="python")


def _write(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep=TAB, index=False)


def _to_int64(x):
    return pd.to_numeric(x, errors="coerce").astype("Int64")


def _q_from_m(m: int) -> str:
    return f"Q{(int(m) - 1) // 3 + 1}"


def _qcode(qstr: str) -> int:
    m = re.match(r"^\s*Q([1-4])\s*$", str(qstr))
    return int(m.group(1)) if m else np.nan


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    return np.where((n > 0) & (d > 0), n / d, np.nan)


def _require_columns(df: pd.DataFrame, cols: List[str], label: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _encode_yq(year: pd.Series, quarter: pd.Series) -> pd.Series:
    return _to_int64(year) * 10 + pd.to_numeric(quarter.map(_qcode), errors="coerce").astype("Int64")


def find_thesis_root(start: Path) -> Path:
    cur = start.resolve()
    for candidate in [cur] + list(cur.parents):
        if (candidate / "Data").exists() and (candidate / "Design").exists():
            return candidate
    raise FileNotFoundError("Could not find thesis root containing both Data/ and Design/.")


# -----------------------------------------------------------------------------
# Monthly port LP: pre-reform port-level object
# -----------------------------------------------------------------------------

def build_monthly_port_raw(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tons_m = _read(args.s1_port_month_tons)
    teu_m = _read(args.s2_port_month_teu)
    lproxy = _read(args.lproxy)

    # Important v4 change:
    # - For labor, we no longer require month_index because the tons-only L_Proxy
    #   uses a different month_index convention than Stage 2 TEU.
    _require_columns(tons_m, ["port", "year", "month", "month_index", "tons_port_m"], "S1_port_month_tons")
    _require_columns(teu_m, ["port", "year", "month", "month_index", "TEU_port_m"], "S2_port_month_teu")
    _require_columns(lproxy, ["port", "year", "month", "L_hours_i_m"], "L_Proxy_v6")

    tons_m = tons_m.copy()
    teu_m = teu_m.copy()
    lproxy = lproxy.copy()

    for df in (tons_m, teu_m, lproxy):
        df["year"] = _to_int64(df["year"])
        df["month"] = _to_int64(df["month"])

    # Aggregate monthly port labor using true calendar keys only.
    labor_p_m = (
        lproxy.groupby(["port", "year", "month"], as_index=False)["L_hours_i_m"]
        .sum(min_count=1)
        .rename(columns={"L_hours_i_m": "L_hours_port_m"})
    )

    # Keep the TEU-side month_index convention, since Stage 2 TEU is the natural
    # owner of the monthly port TEU rows and Model 1A does not rely on month_index
    # for the quarterly collapse anyway.
    teu_m["ym"] = _to_int64(teu_m["year"]) * 100 + _to_int64(teu_m["month"])
    teu_m = teu_m[(teu_m["ym"] >= args.monthly_start) & (teu_m["ym"] <= args.monthly_end)].copy()

    # v4 fix: merge by (port, year, month), not by month_index
    out = (
        teu_m.merge(
            labor_p_m[["port", "year", "month", "L_hours_port_m"]],
            on=["port", "year", "month"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            tons_m[["port", "year", "month", "month_index", "quarter", "tons_port_m", "tons_source"]],
            on=["port", "year", "month"],
            how="left",
            suffixes=("", "_tons"),
            validate="many_to_one",
        )
    )

    # Preserve the TEU-side month_index as the primary month index in the final file.
    out["year"] = _to_int64(out["year"])
    out["month"] = _to_int64(out["month"])
    out["month_index"] = _to_int64(out["month_index"])
    if "quarter" not in out.columns or out["quarter"].isna().all():
        out["quarter"] = out["month"].apply(_q_from_m)

    out["LP_raw"] = _safe_ratio(out["TEU_port_m"], out["L_hours_port_m"])

    keep = [
        "port", "year", "month", "month_index", "quarter",
        "TEU_port_m", "tons_port_m", "tons_source",
        "L_hours_port_m", "LP_raw",
    ]
    out = out[keep].sort_values(["port", "year", "month"]).reset_index(drop=True)

    qa_rows = []
    for p in sorted(out["port"].dropna().unique().tolist()):
        sub = out[out["port"] == p].copy()
        n_lp_na = int(sub["LP_raw"].isna().sum())
        n_labor_na = int(sub["L_hours_port_m"].isna().sum())
        n_rows = len(sub)

        qa_rows.append({
            "check": f"unique_{p}_port_month",
            "ok": bool(not sub.duplicated(["port", "year", "month"]).any()),
            "note": f"n={n_rows}",
        })
        qa_rows.append({
            "check": f"na_monthly_labor_{p}",
            "ok": bool(n_labor_na < n_rows),
            "note": f"{n_labor_na} NA labor rows of {n_rows}",
        })
        qa_rows.append({
            "check": f"na_lp_months_{p}",
            "ok": bool(n_lp_na < n_rows),
            "note": f"{n_lp_na} NA LP rows of {n_rows}",
        })

        # Hard fail if an entire port's monthly LP branch is missing. This is exactly
        # the failure mode that caused the Haifa competition incumbent splice to lose
        # its pre-period support in the v3 run.
        if n_lp_na == n_rows and n_rows > 0:
            raise ValueError(
                f"Monthly port LP is entirely missing for {p}. "
                f"This usually means the port-month labor merge failed. "
                f"Check year/month alignment across S2_port_month_teu and the L_Proxy."
            )

    qa = pd.DataFrame(qa_rows)
    return out, qa


# -----------------------------------------------------------------------------
# Quarterly terminal LP: post-reform terminal-level object
# -----------------------------------------------------------------------------

def build_quarterly_terminal_raw(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    teu_tq = _read(args.s2_term_quarter_teu)
    lproxy = _read(args.lproxy)

    _require_columns(teu_tq, ["port", "terminal", "year", "quarter", "TEU_i_q"], "S2_terminal_quarter_teu")
    _require_columns(lproxy, ["port", "terminal", "year", "month", "L_hours_i_m"], "L_Proxy_v6")

    lproxy = lproxy.copy()
    lproxy["year"] = _to_int64(lproxy["year"])
    lproxy["month"] = _to_int64(lproxy["month"])
    if "quarter" not in lproxy.columns or lproxy["quarter"].isna().all():
        lproxy["quarter"] = lproxy["month"].apply(_q_from_m)

    labor_i_q = (
        lproxy.groupby(["port", "terminal", "year", "quarter"], as_index=False)["L_hours_i_m"]
        .sum(min_count=1)
        .rename(columns={"L_hours_i_m": "L_hours_i_q"})
    )

    out = teu_tq.merge(labor_i_q, on=["port", "terminal", "year", "quarter"], how="left")

    if args.s1_terminal_quarter_tons and Path(args.s1_terminal_quarter_tons).exists():
        t_tq = _read(args.s1_terminal_quarter_tons)
        if {"port", "terminal", "year", "quarter", "tons_i_q"}.issubset(set(t_tq.columns)):
            out = out.merge(
                t_tq[["port", "terminal", "year", "quarter", "tons_i_q"]],
                on=["port", "terminal", "year", "quarter"],
                how="left",
            )
        else:
            out["tons_i_q"] = np.nan
    else:
        out["tons_i_q"] = np.nan

    out["LP_raw"] = _safe_ratio(out["TEU_i_q"], out["L_hours_i_q"])

    out["yq"] = _encode_yq(out["year"], out["quarter"])
    qs = int(str(args.quarterly_start)[:4]) * 10 + int(str(args.quarterly_start)[-1])
    qe = int(str(args.quarterly_end)[:4]) * 10 + int(str(args.quarterly_end)[-1])
    out = out[(out["yq"] >= qs) & (out["yq"] <= qe)].copy()

    keep = ["port", "terminal", "year", "quarter", "TEU_i_q", "tons_i_q", "L_hours_i_q", "LP_raw"]
    out = out[keep].sort_values(["port", "terminal", "year", "quarter"]).reset_index(drop=True)

    qa_rows = [{
        "check": "unique_terminal_quarter",
        "ok": bool(not out.duplicated(["port", "terminal", "year", "quarter"]).any()),
        "note": f"n={len(out)}",
    }]

    for term in ["Haifa-Legacy", "Haifa-Bayport", "Ashdod-Legacy", "Ashdod-HCT"]:
        sub = out[out["terminal"] == term].copy()
        qa_rows.append({
            "check": f"na_lp_quarters_{term}",
            "ok": True,
            "note": f"{int(sub['LP_raw'].isna().sum())} NA of {len(sub)}",
        })

    qa = pd.DataFrame(qa_rows)
    return out, qa


# -----------------------------------------------------------------------------
# Stack panel in the same mixed-frequency schema used by Model 1A
# -----------------------------------------------------------------------------

def write_component_files(monthly_port: pd.DataFrame, quarterly_term: pd.DataFrame, out_dir: str):
    _write(monthly_port[monthly_port["port"] == "Haifa"].copy(), os.path.join(out_dir, "LP_Haifa_port_month.tsv"))
    _write(monthly_port[monthly_port["port"] == "Ashdod"].copy(), os.path.join(out_dir, "LP_Ashdod_port_month.tsv"))

    mapping = {
        "Haifa-Legacy": "LP_Haifa_Legacy_quarter.tsv",
        "Haifa-Bayport": "LP_Haifa_SIPG_quarter.tsv",
        "Ashdod-Legacy": "LP_Ashdod_Legacy_quarter.tsv",
        "Ashdod-HCT": "LP_Ashdod_HCT_quarter.tsv",
    }
    for term, fname in mapping.items():
        _write(quarterly_term[quarterly_term["terminal"] == term].copy(), os.path.join(out_dir, fname))


def stack_panel(monthly_port: pd.DataFrame, quarterly_term: pd.DataFrame) -> pd.DataFrame:
    hm = monthly_port[monthly_port["port"] == "Haifa"].copy().reset_index(drop=True)
    am = monthly_port[monthly_port["port"] == "Ashdod"].copy().reset_index(drop=True)

    def _monthly_transform(df: pd.DataFrame, series_id: str, port_name: str) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        idx = df.index
        out = pd.DataFrame({
            "series_id": pd.Series([series_id] * len(df), index=idx),
            "level": pd.Series(["port"] * len(df), index=idx),
            "freq": pd.Series(["M"] * len(df), index=idx),
            "port": pd.Series([port_name] * len(df), index=idx),
            "terminal": pd.Series([pd.NA] * len(df), index=idx),
            "year": _to_int64(df["year"]),
            "month": _to_int64(df["month"]),
            "quarter": df["quarter"],
            "month_index": _to_int64(df["month_index"]),
            "quarter_index": pd.Series([pd.NA] * len(df), dtype="Int64", index=idx),
            "TEU": df["TEU_port_m"],
            "tons": df["tons_port_m"],
            "L_hours": df["L_hours_port_m"],
            "w": pd.Series([1.0] * len(df), index=idx),
            "Pi": pd.Series([pd.NA] * len(df), index=idx),
            "LP": df["LP_raw"],
            "LP_id": df["LP_raw"],
            "tons_source": df["tons_source"] if "tons_source" in df.columns else pd.Series([pd.NA] * len(df), index=idx),
            "lp_definition": pd.Series(["raw_teu_over_labor"] * len(df), index=idx),
        })
        return out.reset_index(drop=True)

    def _quarterly_transform(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        idx = df.index
        qidx = _to_int64(df["year"]) * 4 + pd.to_numeric(df["quarter"].map(_qcode), errors="coerce").astype("Int64")
        out = pd.DataFrame({
            "series_id": pd.Series([series_id] * len(df), index=idx),
            "level": pd.Series(["terminal"] * len(df), index=idx),
            "freq": pd.Series(["Q"] * len(df), index=idx),
            "port": df["port"],
            "terminal": df["terminal"],
            "year": _to_int64(df["year"]),
            "month": pd.Series([pd.NA] * len(df), dtype="Int64", index=idx),
            "quarter": df["quarter"],
            "month_index": pd.Series([pd.NA] * len(df), dtype="Int64", index=idx),
            "quarter_index": qidx,
            "TEU": df["TEU_i_q"],
            "tons": df["tons_i_q"],
            "L_hours": df["L_hours_i_q"],
            "w": pd.Series([1.0] * len(df), index=idx),
            "Pi": pd.Series([pd.NA] * len(df), index=idx),
            "LP": df["LP_raw"],
            "LP_id": df["LP_raw"],
            "tons_source": pd.Series([pd.NA] * len(df), index=idx),
            "lp_definition": pd.Series(["raw_teu_over_labor"] * len(df), index=idx),
        })
        return out.reset_index(drop=True)

    pieces = [
        _monthly_transform(hm, "Haifa_port_M", "Haifa"),
        _monthly_transform(am, "Ashdod_port_M", "Ashdod"),
        _quarterly_transform(quarterly_term[quarterly_term["terminal"] == "Haifa-Legacy"].copy(), "Haifa_Legacy_Q"),
        _quarterly_transform(quarterly_term[quarterly_term["terminal"] == "Haifa-Bayport"].copy(), "Haifa_SIPG_Q"),
        _quarterly_transform(quarterly_term[quarterly_term["terminal"] == "Ashdod-Legacy"].copy(), "Ashdod_Legacy_Q"),
        _quarterly_transform(quarterly_term[quarterly_term["terminal"] == "Ashdod-HCT"].copy(), "Ashdod_HCT_Q"),
    ]

    cols = [
        "series_id", "level", "freq", "port", "terminal",
        "year", "month", "quarter", "month_index", "quarter_index",
        "TEU", "tons", "L_hours", "w", "Pi", "LP", "LP_id", "tons_source", "lp_definition",
    ]
    panel = pd.concat(pieces, ignore_index=True)[cols]
    return panel.sort_values(["series_id", "year", "month", "quarter"]).reset_index(drop=True)


def build_qa(panel: pd.DataFrame, qa_month: pd.DataFrame, qa_quarter: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(check: str, ok: bool, note: str):
        rows.append({"check": check, "ok": bool(ok), "note": note})

    rows.extend(qa_month.to_dict(orient="records"))
    rows.extend(qa_quarter.to_dict(orient="records"))

    m = panel[panel["freq"] == "M"].copy()
    q = panel[panel["freq"] == "Q"].copy()

    add("unique_monthly_series", not m.duplicated(["series_id", "year", "month"]).any(), f"n={len(m)}")
    add("unique_quarterly_series", not q.duplicated(["series_id", "year", "quarter"]).any(), f"n={len(q)}")

    for sid, grp in panel.groupby("series_id"):
        add(
            f"na_rates_{sid}",
            True,
            f"NA TEU={int(grp['TEU'].isna().sum())}, NA L_hours={int(grp['L_hours'].isna().sum())}, NA LP={int(grp['LP'].isna().sum())}"
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    thesis_root = find_thesis_root(script_dir)

    parser.add_argument("--s1_port_month_tons", default=str(thesis_root / "Data" / "LP" / "S1_port_month_tons.tsv"))
    parser.add_argument("--s1_terminal_quarter_tons", default=str(thesis_root / "Data" / "LP" / "S1_terminal_quarter_tons.tsv"))
    parser.add_argument("--s2_port_month_teu", default=str(thesis_root / "Data" / "LP" / "S2_port_month_teu.tsv"))
    parser.add_argument("--s2_term_quarter_teu", default=str(thesis_root / "Data" / "LP" / "S2_terminal_quarter_teu.tsv"))
    parser.add_argument("--lproxy", default=str(thesis_root / "Data" / "L_proxy" / "common_rule_v6_tonsonly" / "L_Proxy_commonrule_v6_tonsonly.tsv"))
    parser.add_argument("--out", default=str(thesis_root / "Data" / "LP" / "raw_from_l_v4_tonsonly"))
    parser.add_argument("--monthly_start", type=int, default=201801)
    parser.add_argument("--monthly_end", type=int, default=202110)
    parser.add_argument("--quarterly_start", type=str, default="2021Q3")
    parser.add_argument("--quarterly_end", type=str, default="2024Q4")
    parser.add_argument("--also-write-canonical", action="store_true", help="Also overwrite Data/LP/LP_Panel.tsv with the new build")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    monthly_port, qa_m = build_monthly_port_raw(args)
    quarterly_term, qa_q = build_quarterly_terminal_raw(args)
    write_component_files(monthly_port, quarterly_term, args.out)

    panel = stack_panel(monthly_port, quarterly_term)
    qa = build_qa(panel, qa_m, qa_q)

    panel_path = os.path.join(args.out, "LP_Panel.tsv")
    qa_path = os.path.join(args.out, "LP_raw_qa.tsv")
    _write(panel, panel_path)
    _write(qa, qa_path)

    meta = {
        "builder": "Build_LP_Panel_raw_from_L_v4_tonsonly.py",
        "definition": {
            "monthly_port": "LP_{p,m} = TEU_{p,m} / L_{p,m}",
            "quarterly_terminal": "LP_{i,q} = TEU_{i,q} / L_{i,q}",
            "quarterly_labor": "L_{i,q} = sum_{m in q} L_{i,m}",
        },
        "why_v4": [
            "fixes the monthly port labor merge by joining on (port, year, month) rather than month_index",
            "keeps the quarterly terminal LP branch unchanged",
            "adds hard QA failure if a port's monthly LP branch is entirely missing",
        ],
        "inputs": {
            "s1_port_month_tons": args.s1_port_month_tons,
            "s1_terminal_quarter_tons": args.s1_terminal_quarter_tons,
            "s2_port_month_teu": args.s2_port_month_teu,
            "s2_term_quarter_teu": args.s2_term_quarter_teu,
            "lproxy": args.lproxy,
        },
        "outputs": {
            "LP_Panel": panel_path,
            "LP_raw_qa": qa_path,
        },
    }

    meta_path = os.path.join(args.out, "_meta_lp_raw_v4_tonsonly.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if args.also_write_canonical:
        canonical = thesis_root / "Data" / "LP" / "LP_Panel.tsv"
        _write(panel, str(canonical))
        meta["outputs"]["canonical_LP_Panel"] = str(canonical)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print("[LP_raw_v4_tonsonly] Wrote raw throughput-over-labor LP artifacts to", args.out)


if __name__ == "__main__":
    main()
