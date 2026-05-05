#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build_LP_Panel_raw_from_L_v2.py

Patch note
----------
v1 built the six component raw-LP files correctly, but its stacked LP_Panel.tsv
could contain duplicate/blank rows because some sliced DataFrames kept their
original indices and were combined with placeholder Series on fresh RangeIndex
objects. Pandas aligned on the union of indices, which produced extra all-NA
rows for several series.

v2 fixes that by resetting indices before each transform and by constructing
placeholder columns with the same index as the input slice.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

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


def _require_columns(df: pd.DataFrame, cols: list[str], label: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _encode_yq(year: pd.Series, quarter: pd.Series) -> pd.Series:
    return _to_int64(year) * 10 + pd.to_numeric(quarter.map(_qcode), errors="coerce").astype("Int64")


def build_monthly_port_raw(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    tons_m = _read(args.s1_port_month_tons)
    teu_m = _read(args.s2_port_month_teu)
    labor_p_m = _read(args.s3_port_month_labor)

    _require_columns(tons_m, ["port", "year", "month", "month_index", "tons_port_m"], "S1_port_month_tons")
    _require_columns(teu_m, ["port", "year", "month", "month_index", "TEU_port_m"], "S2_port_month_teu")
    _require_columns(labor_p_m, ["port", "year", "month", "month_index", "L_hours_port_m"], "S3_port_month_labor")

    teu_m = teu_m.copy()
    teu_m["ym"] = _to_int64(teu_m["year"]) * 100 + _to_int64(teu_m["month"])
    teu_m = teu_m[(teu_m["ym"] >= args.monthly_start) & (teu_m["ym"] <= args.monthly_end)].copy()

    out = (
        teu_m.merge(
            labor_p_m[["port", "year", "month", "month_index", "L_hours_port_m"]],
            on=["port", "year", "month", "month_index"],
            how="left",
        )
        .merge(
            tons_m[["port", "year", "month", "month_index", "quarter", "tons_port_m", "tons_source"]],
            on=["port", "year", "month", "month_index"],
            how="left",
        )
    )

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
        qa_rows.append({
            "check": f"unique_{p}_port_month",
            "ok": bool(not sub.duplicated(["port", "year", "month"]).any()),
            "note": f"n={len(sub)}",
        })
        qa_rows.append({
            "check": f"na_lp_months_{p}",
            "ok": True,
            "note": f"{int(sub['LP_raw'].isna().sum())} NA of {len(sub)}",
        })

    qa = pd.DataFrame(qa_rows)
    return out, qa


def build_quarterly_terminal_raw(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    teu_tq = _read(args.s2_term_quarter_teu)
    lproxy = _read(args.s3_lproxy_clean)

    _require_columns(teu_tq, ["port", "terminal", "year", "quarter", "TEU_i_q"], "S2_terminal_quarter_teu")
    _require_columns(lproxy, ["port", "terminal", "year", "month", "L_hours_i_m"], "S3_lproxy_clean")

    lproxy = lproxy.copy()
    lproxy["year"] = _to_int64(lproxy["year"])
    lproxy["month"] = _to_int64(lproxy["month"])
    if "quarter" not in lproxy.columns or lproxy["quarter"].isna().all():
        lproxy["quarter"] = lproxy["month"].apply(_q_from_m)

    labor_i_q = (
        lproxy.groupby(["port", "terminal", "year", "quarter"], as_index=False)["L_hours_i_m"]
        .sum()
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

    keep = [
        "port", "terminal", "year", "quarter",
        "TEU_i_q", "tons_i_q", "L_hours_i_q", "LP_raw",
    ]
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


def write_component_files(monthly_port: pd.DataFrame, quarterly_term: pd.DataFrame, out_dir: str):
    _write(monthly_port[monthly_port["port"] == "Haifa"].copy(),
           os.path.join(out_dir, "LP_Haifa_port_month.tsv"))
    _write(monthly_port[monthly_port["port"] == "Ashdod"].copy(),
           os.path.join(out_dir, "LP_Ashdod_port_month.tsv"))

    mapping = {
        "Haifa-Legacy": "LP_Haifa_Legacy_quarter.tsv",
        "Haifa-Bayport": "LP_Haifa_SIPG_quarter.tsv",
        "Ashdod-Legacy": "LP_Ashdod_Legacy_quarter.tsv",
        "Ashdod-HCT": "LP_Ashdod_HCT_quarter.tsv",
    }
    for term, fname in mapping.items():
        _write(quarterly_term[quarterly_term["terminal"] == term].copy(),
               os.path.join(out_dir, fname))


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
        "TEU", "tons", "L_hours", "w", "Pi", "LP", "LP_id", "tons_source",
        "lp_definition",
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1_port_month_tons", default="Data/LP/S1_port_month_tons.tsv")
    ap.add_argument("--s1_terminal_quarter_tons", default="Data/LP/S1_terminal_quarter_tons.tsv")
    ap.add_argument("--s2_port_month_teu", default="Data/LP/S2_port_month_teu.tsv")
    ap.add_argument("--s2_term_quarter_teu", default="Data/LP/S2_terminal_quarter_teu.tsv")
    ap.add_argument("--s3_lproxy_clean", default="Data/LP/S3_lproxy_clean.tsv")
    ap.add_argument("--s3_port_month_labor", default="Data/LP/S3_port_month_labor.tsv")
    ap.add_argument("--out", default="Data/LP/raw_from_l_v2")
    ap.add_argument("--monthly_start", type=int, default=201801)
    ap.add_argument("--monthly_end", type=int, default=202110)
    ap.add_argument("--quarterly_start", type=str, default="2021Q3")
    ap.add_argument("--quarterly_end", type=str, default="2024Q4")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    monthly_port, qa_m = build_monthly_port_raw(args)
    quarterly_term, qa_q = build_quarterly_terminal_raw(args)

    write_component_files(monthly_port, quarterly_term, args.out)

    panel = stack_panel(monthly_port, quarterly_term)
    qa = build_qa(panel, qa_m, qa_q)

    _write(panel, os.path.join(args.out, "LP_Panel.tsv"))
    _write(qa, os.path.join(args.out, "LP_raw_qa.tsv"))

    meta = {
        "builder": "Build_LP_Panel_raw_from_L_v2.py",
        "definition": {
            "monthly_port": "LP_{p,m} = TEU_{p,m} / L_{p,m}",
            "quarterly_terminal": "LP_{i,q} = TEU_{i,q} / L_{i,q}",
            "quarterly_labor": "L_{i,q} = sum_{m in q} L_{i,m}",
        },
        "outputs": {
            "LP_Panel": os.path.join(args.out, "LP_Panel.tsv"),
            "LP_raw_qa": os.path.join(args.out, "LP_raw_qa.tsv"),
        },
        "patch_note": "Reset indices before stack transforms so the stacked panel has no duplicated blank rows.",
    }

    with open(os.path.join(args.out, "_meta_lp_raw.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[LP_raw_v2] Wrote raw throughput-over-labor LP artifacts to", args.out)


if __name__ == "__main__":
    main()
