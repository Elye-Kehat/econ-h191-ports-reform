#!/usr/bin/env python3
"""
Model_2_to_tables(v3)_final2.py

Format corrected Model 2 accounting outputs into table-ready TSV and simple TeX helpers.

This file is presentation-only. It does not estimate any regressions and does not recompute
Model 2 accounting except for formatting already-computed columns.

Outputs
-------
Design/Output (new)/Model_2/Tables/
  - model2_table5_preferred.tsv
  - model2_table6_detailed.tsv
  - model2_table7_elasticity_robustness.tsv
  - model2_table5_preferred.tex
  - model2_table6_detailed.tex
  - model2_table7_elasticity_robustness.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def find_thesis_root(start: Optional[Path] = None) -> Path:
    here = start or Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "Data").exists() and (p / "Design").exists():
            return p
    raise FileNotFoundError("Could not locate thesis root. Run this from inside the Thesis project.")


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path, sep="\t")


def require_manifest_ok(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing accounting manifest: {path}. Run Model_2_accounting(v3)_final2.py first.")


def fmt_num(x: object, digits: int = 3, blank: str = "--") -> str:
    try:
        v = float(x)
    except Exception:
        return blank
    if not np.isfinite(v):
        return blank
    return f"{v:.{digits}f}"


def fmt_share(x: object, blank: str = "--") -> str:
    try:
        v = float(x)
    except Exception:
        return blank
    if not np.isfinite(v):
        return blank
    return f"{100 * v:.1f}%"


def tex_escape(s: object) -> str:
    if pd.isna(s):
        return ""
    text = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b in repl.items():
        text = text.replace(a, b)
    return text


def row_label(row: pd.Series) -> str:
    if "row_label" in row.index and pd.notna(row["row_label"]):
        return str(row["row_label"])
    labels = {
        "competition_legacy": "Competition - Legacy",
        "competition_aggregate": "Competition - Aggregate",
        "privatization_legacy": "Privatization - Legacy",
        "privatization_aggregate": "Privatization - Aggregate",
    }
    return labels.get(str(row.get("row_key", "")), str(row.get("row_key", "")))


def build_preferred_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = ["competition_legacy", "competition_aggregate", "privatization_legacy", "privatization_aggregate"]
    d = df.copy()
    d["_order"] = d["row_key"].map({k: i for i, k in enumerate(order)}).fillna(99)
    d = d.sort_values(["_order", "row_key"])

    for _, r in d.iterrows():
        ok = str(r.get("status", "")).lower() == "ok"
        rows.append({
            "row": row_label(r),
            "horizon": r.get("horizon", ""),
            "TE": fmt_num(r.get("TE"), 3),
            "dC": fmt_num(r.get("dC"), 3) if ok else "--",
            "eta": fmt_num(r.get("eta"), 3) if ok else "--",
            "CD": fmt_num(r.get("CD"), 3) if ok else "--",
            "share_explained": fmt_share(r.get("share_explained")) if ok else "--",
            "share_valid": str(bool(r.get("share_valid", False))) if ok else "False",
            "status": r.get("status", ""),
            "reason": r.get("reason", ""),
        })
    return pd.DataFrame(rows)


def build_detailed_table(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "row_key", "row_label", "reform", "entity", "estimator_family", "horizon",
        "TE", "TE_se", "dC", "dC_se", "eta", "eta_source", "CD", "residual",
        "share_explained", "sign_consistent", "share_valid", "spec_te", "spec_dC", "status", "reason",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    if "row_label" not in out.columns:
        out["row_label"] = df.apply(row_label, axis=1)
    return out


def build_robustness_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        ok = str(r.get("status", "")).lower() == "ok"
        rows.append({
            "row": row_label(r),
            "eta_family": r.get("eta_family", ""),
            "eta_source": r.get("eta_source", ""),
            "eta_role": r.get("eta_role", ""),
            "eta_spec": r.get("eta_spec", ""),
            "eta": fmt_num(r.get("eta"), 3) if str(r.get("status", "")).lower() == "ok" else "--",
            "TE": fmt_num(r.get("TE"), 3),
            "dC": fmt_num(r.get("dC"), 3) if ok else "--",
            "CD": fmt_num(r.get("CD"), 3) if ok else "--",
            "share_explained": fmt_share(r.get("share_explained")) if ok else "--",
            "share_valid": str(bool(r.get("share_valid", False))) if ok else "False",
            "status": r.get("status", ""),
        })
    return pd.DataFrame(rows)


def write_simple_tex(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    cols = list(df.columns)
    align = "l" + "c" * (len(cols) - 1)
    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{{tex_escape(caption)}}}")
    lines.append(f"  \\label{{{tex_escape(label)}}}")
    lines.append(r"  \scriptsize")
    lines.append(r"  \begin{tabular}{" + align + r"}")
    lines.append(r"    \toprule")
    lines.append("    " + " & ".join(tex_escape(c) for c in cols) + " " + chr(92) * 2)
    lines.append(r"    \midrule")
    for _, r in df.iterrows():
        lines.append("    " + " & ".join(tex_escape(r[c]) for c in cols) + " " + chr(92) * 2)
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    thesis_root = find_thesis_root()
    parser = argparse.ArgumentParser(description="Format corrected Model 2 accounting outputs into table helpers.")
    parser.add_argument(
        "--preferred",
        type=Path,
        default=thesis_root / "Design" / "Output (new)" / "Model_2" / "Tables" / "model2_accounting_preferred.tsv",
    )
    parser.add_argument(
        "--detailed",
        type=Path,
        default=thesis_root / "Design" / "Output (new)" / "Model_2" / "Tables" / "model2_accounting_long.tsv",
    )
    parser.add_argument(
        "--robustness",
        type=Path,
        default=thesis_root / "Design" / "Output (new)" / "Model_2" / "Tables" / "model2_accounting_elasticity_robustness.tsv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=thesis_root / "Design" / "Output (new)" / "Model_2" / "Tables",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=thesis_root / "Design" / "Output (new)" / "Model_2" / "Tables" / "model2_accounting_manifest.json",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    require_manifest_ok(args.manifest)

    preferred_in = read_tsv(args.preferred)
    detailed_in = read_tsv(args.detailed)
    robustness_in = read_tsv(args.robustness) if args.robustness.exists() else pd.DataFrame()

    table5 = build_preferred_table(preferred_in)
    table6 = build_detailed_table(detailed_in)
    table7 = build_robustness_table(robustness_in) if not robustness_in.empty else pd.DataFrame()

    table5_path = args.outdir / "model2_table5_preferred.tsv"
    table6_path = args.outdir / "model2_table6_detailed.tsv"
    table7_path = args.outdir / "model2_table7_elasticity_robustness.tsv"
    table5_tex = args.outdir / "model2_table5_preferred.tex"
    table6_tex = args.outdir / "model2_table6_detailed.tex"
    table7_tex = args.outdir / "model2_table7_elasticity_robustness.tex"

    table5.to_csv(table5_path, sep="\t", index=False)
    table6.to_csv(table6_path, sep="\t", index=False)
    table7.to_csv(table7_path, sep="\t", index=False)

    write_simple_tex(
        table5,
        table5_tex,
        caption="Model 2 preferred accounting decomposition",
        label="tab:model2_preferred_accounting",
    )
    write_simple_tex(
        table6,
        table6_tex,
        caption="Model 2 detailed accounting decomposition helper",
        label="tab:model2_detailed_accounting_helper",
    )
    if not table7.empty:
        write_simple_tex(
            table7,
            table7_tex,
            caption="Model 2 elasticity robustness decomposition",
            label="tab:model2_elasticity_robustness",
        )

    print("=== Model_2_to_tables(v3)_final2: done ===")
    print(f"Manifest      : {args.manifest}")
    print(f"Preferred in  : {args.preferred}")
    print(f"Detailed in   : {args.detailed}")
    print(f"Robustness in : {args.robustness}")
    print(f"Table 5 TSV   : {table5_path}")
    print(f"Table 6 TSV   : {table6_path}")
    print(f"Table 7 TSV   : {table7_path}")
    print(f"Table 5 TeX   : {table5_tex}")
    print(f"Table 6 TeX   : {table6_tex}")
    print(f"Table 7 TeX   : {table7_tex if not table7.empty else '(not written, robustness input empty)'}")


if __name__ == "__main__":
    main()
