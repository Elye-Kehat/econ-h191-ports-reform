#!/usr/bin/env python3
"""
Model_2_step4_to_tables_v13.py

Format the enhanced Model 2 accounting outputs into thesis-facing tables and
companion diagnostics.

v13 changes
-----------
1. Keeps the existing summary and diagnostics outputs.
2. Replaces the appendix table writer with a template-aware mapper that fills
   the user's fixed LaTeX appendix layout from the actual supported Model 2
   output space.
3. Fixes avoidable missing cells caused by template/output mismatches.
4. Preserves horizons/windows exactly as produced upstream; cells remain '--'
   only when no true supporting row exists at that horizon.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------

def find_thesis_root(start: Optional[Path] = None) -> Path:
    here = start or Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "Data").exists() and (p / "Design").exists():
            return p
    raise FileNotFoundError("Could not locate thesis root.")


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path, sep="\t")


def require_manifest_ok(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing accounting manifest: {path}. Run Model_2_step3_accounting_v12.py first.")


def fmt_num(x: object, digits: int = 3, blank: str = "--") -> str:
    try:
        v = float(x)
    except Exception:
        return blank
    if not np.isfinite(v):
        return blank
    return f"{v:.{digits}f}"


def fmt_share_decimal(x: object, digits: int = 3, blank: str = "--") -> str:
    try:
        v = float(x)
    except Exception:
        return blank
    if not np.isfinite(v):
        return blank
    return f"{v:.{digits}f}"


def fmt_share_pct(x: object, blank: str = "--") -> str:
    try:
        v = float(x)
    except Exception:
        return blank
    if not np.isfinite(v):
        return blank
    return f"{100 * v:.1f}%"


def fmt_ci(lo: object, hi: object, digits: int = 3, blank: str = "--") -> str:
    try:
        lo_v = float(lo)
        hi_v = float(hi)
    except Exception:
        return blank
    if not (np.isfinite(lo_v) and np.isfinite(hi_v)):
        return blank
    return f"[{lo_v:.{digits}f}, {hi_v:.{digits}f}]"


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


def is_close(x: object, target: float, tol: float = 1e-6) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return np.isfinite(v) and abs(v - target) <= tol


# ---------------------------------------------------------------------
# Existing generic tables
# ---------------------------------------------------------------------

def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        ok = str(r.get("status", "")).lower() == "ok"
        eta_class = str(r.get("eta_interpretation_class", ""))
        if eta_class == "preferred_manual":
            eta_note = "Preferred manual"
        elif eta_class == "preferred_manual_aggregate":
            eta_note = "Preferred manual aggregate proxy"
        elif eta_class == "aggregate_regression_fallback":
            eta_note = "Aggregate regression fallback"
        else:
            eta_note = eta_class
        rows.append({
            "row": r.get("row_label", r.get("row_key", "")),
            "post_window": r.get("horizon", ""),
            "lp_input": f"{r.get('lp_family', '')} / {r.get('lp_spec', '')}" if ok else "--",
            "kl_input": f"{r.get('kl_spec', '')}" if ok else "--",
            "eta_note": eta_note if ok else "--",
            "TE": fmt_num(r.get("TE"), 3),
            "dC": fmt_num(r.get("dC"), 3) if ok else "--",
            "eta": fmt_num(r.get("eta"), 3) if ok else "--",
            "CD": fmt_num(r.get("CD"), 3) if ok else "--",
            "share_explained": fmt_share_pct(r.get("share_explained")) if ok else "--",
        })
    return pd.DataFrame(rows)


def build_diagnostics_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "row": r.get("row_label", r.get("row_key", "")),
            "post_window": r.get("horizon", ""),
            "TE_ci": fmt_ci(r.get("TE_ci_lo"), r.get("TE_ci_hi")),
            "dC_ci": fmt_ci(r.get("dC_ci_lo"), r.get("dC_ci_hi")),
            "CD_ci": fmt_ci(r.get("CD_ci_lo"), r.get("CD_ci_hi")),
            "share_ci": fmt_ci(r.get("share_ci_lo"), r.get("share_ci_hi")),
            "TE_R2": fmt_num(r.get("TE_R2"), 3),
            "eta_class": r.get("eta_interpretation_class", ""),
            "sign_consistent": r.get("sign_consistent", ""),
            "share_valid": r.get("share_valid", ""),
            "warning_flags": r.get("warning_flags", r.get("warnings", "")),
            "kl_components": r.get("kl_component_windows", ""),
            "kl_weights": r.get("kl_component_month_weights", ""),
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
    lines.append("    " + " & ".join(tex_escape(c) for c in cols) + r" \\")
    lines.append(r"    \midrule")
    for _, row in df.iterrows():
        lines.append("    " + " & ".join(tex_escape(row[c]) for c in cols) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# Appendix template-aware mapping
# ---------------------------------------------------------------------

def normalize_appendix_sources(appendix_df: pd.DataFrame, diag_all_df: pd.DataFrame) -> pd.DataFrame:
    app = appendix_df.copy()
    dia = diag_all_df.copy()

    if not app.empty:
        join_cols = [c for c in ["row_key", "horizon", "lp_family", "lp_spec", "kl_spec", "eta", "TE", "dC"] if c in app.columns and c in dia.columns]
        if join_cols:
            dia_cols = [c for c in dia.columns if c not in join_cols]
            app = app.merge(dia[join_cols + dia_cols], on=join_cols, how="left")
        else:
            for c in dia.columns:
                if c not in app.columns:
                    app[c] = np.nan
        return app

    return dia


def infer_eta_label(row: pd.Series) -> str:
    eta_class = str(row.get("eta_interpretation_class", ""))
    eta = row.get("eta", np.nan)

    if eta_class == "preferred_manual":
        return f"Manual (HPC labor share; {fmt_num(eta)})"
    if eta_class == "preferred_manual_aggregate":
        return f"Manual (HPC+IPC 2024 proxy; {fmt_num(eta)})"
    if eta_class == "aggregate_regression_fallback":
        return f"Regression (cluster TS+trend; {fmt_num(eta)})"
    if eta_class == "regression_robustness":
        return f"Regression (legacy TS+trend; {fmt_num(eta)})"
    if eta_class == "manual_robustness":
        if is_close(eta, 1/3, 5e-4):
            return f"Manual (one-third benchmark; {fmt_num(eta)})"
        if is_close(eta, 0.42, 5e-4):
            return f"Manual (Hazan--Tsur benchmark; {fmt_num(eta)})"
        return f"Manual robustness ({fmt_num(eta)})"
    return f"--"


def select_row(df: pd.DataFrame, **criteria) -> Optional[pd.Series]:
    sub = df.copy()
    for k, v in criteria.items():
        if k not in sub.columns:
            return None
        if isinstance(v, (list, tuple, set)):
            sub = sub[sub[k].isin(list(v))]
        else:
            sub = sub[sub[k] == v]
    if sub.empty:
        return None
    # deterministic order: baseline before +Tr already upstream, but sort defensively
    sort_cols = [c for c in ["TE_se", "dC_se", "TE_R2"] if c in sub.columns]
    if sort_cols:
        ascending = [True, True, False][:len(sort_cols)]
        sub = sub.sort_values(sort_cols, ascending=ascending)
    return sub.iloc[0]


def display_row(slot_label: str, r: Optional[pd.Series], default_window: str = "--",
                default_lp_family: str = "--", default_lp_spec: str = "--",
                default_kl_spec: str = "--", default_kl_variant: str = "--",
                default_eta_input: str = "--") -> Dict[str, str]:
    if r is None:
        return {
            "row_group": slot_label,
            "post_window": default_window,
            "lp_family": default_lp_family,
            "lp_spec": default_lp_spec,
            "kl_spec": default_kl_spec,
            "kl_variant": default_kl_variant,
            "eta_input": default_eta_input,
            "TE": "--",
            "dC": "--",
            "eta": "--",
            "CD": "--",
            "share_explained": "--",
        }
    return {
        "row_group": slot_label,
        "post_window": str(r.get("horizon", default_window)),
        "lp_family": str(r.get("lp_family", default_lp_family)),
        "lp_spec": str(r.get("lp_spec", default_lp_spec)),
        "kl_spec": str(r.get("kl_spec", default_kl_spec)),
        "kl_variant": default_kl_variant,
        "eta_input": infer_eta_label(r),
        "TE": fmt_num(r.get("TE"), 3),
        "dC": fmt_num(r.get("dC"), 3),
        "eta": fmt_num(r.get("eta"), 3),
        "CD": fmt_num(r.get("CD"), 3),
        "share_explained": fmt_share_decimal(r.get("share_explained"), 3),
    }


def build_appendix_template_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []

    # Panel A: Competition - Legacy
    rows.append({"panel": "Panel A: Haifa competition (Legacy)", **display_row(
        "Haifa competition - Legacy",
        select_row(df, row_key="competition_legacy", lp_family="Conventional DiD", lp_spec="Baseline",
                   kl_spec="Baseline", eta_interpretation_class="preferred_manual"),
        default_window="[1,13]", default_kl_variant="Detailed"
    )})
    rows.append({"panel": "Panel A: Haifa competition (Legacy)", **display_row(
        "",
        select_row(df, row_key="competition_legacy", lp_family="Conventional DiD", lp_spec="+Tr",
                   kl_spec="Controls+Trend", eta_interpretation_class="preferred_manual"),
        default_window="[1,13]", default_kl_variant="Detailed"
    )})
    rows.append({"panel": "Panel A: Haifa competition (Legacy)", **display_row(
        "",
        select_row(df, row_key="competition_legacy", lp_family="Conventional DiD", lp_spec="Baseline",
                   kl_spec="Baseline", eta_interpretation_class="regression_robustness"),
        default_window="[1,13]", default_kl_variant="Detailed"
    )})
    rows.append({"panel": "Panel A: Haifa competition (Legacy)", **display_row(
        "",
        select_row(df, row_key="competition_legacy", lp_family="Conventional DiD", lp_spec="+Tr",
                   kl_spec="Controls+Trend", eta_interpretation_class="regression_robustness"),
        default_window="[1,13]", default_kl_variant="Detailed"
    )})
    one_third = df[(df.get("row_key") == "competition_legacy") &
                   (df.get("lp_family") == "Conventional DiD") &
                   (df.get("lp_spec") == "Baseline") &
                   (df.get("kl_spec") == "Baseline") &
                   (df.get("eta_interpretation_class") == "manual_robustness")]
    one_third = one_third[one_third["eta"].apply(lambda x: is_close(x, 1/3, 5e-4))]
    one_third_row = one_third.iloc[0] if not one_third.empty else None
    rows.append({"panel": "Panel A: Haifa competition (Legacy)", **display_row(
        "",
        one_third_row,
        default_window="[1,13]", default_kl_variant="Detailed"
    )})
    # Hazan-Tsur benchmark
    hazan = df[(df.get("row_key") == "competition_legacy") &
               (df.get("lp_family") == "Conventional DiD") &
               (df.get("lp_spec") == "Baseline") &
               (df.get("kl_spec") == "Baseline") &
               (df.get("eta_interpretation_class") == "manual_robustness")]
    hazan = hazan[hazan["eta"].apply(lambda x: is_close(x, 0.42, 5e-4))]
    hazan_row = hazan.iloc[0] if not hazan.empty else None
    rows.append({"panel": "Panel A: Haifa competition (Legacy)", **display_row(
        "",
        hazan_row,
        default_window="[1,13]", default_kl_variant="Detailed"
    )})

    # Panel B: Competition - Aggregate
    rows.append({"panel": "Panel B: Haifa competition (Aggregate)", **display_row(
        "Haifa competition - Aggregate",
        select_row(df, row_key="competition_aggregate", lp_family="Conventional DiD", lp_spec="Baseline",
                   kl_spec="Baseline", eta_interpretation_class="preferred_manual_aggregate"),
        default_window="[1,13]", default_kl_variant="Detailed"
    )})
    rows.append({"panel": "Panel B: Haifa competition (Aggregate)", **display_row(
        "",
        select_row(df, row_key="competition_aggregate", lp_family="Conventional DiD", lp_spec="+Tr",
                   kl_spec="Controls+Trend", eta_interpretation_class="preferred_manual_aggregate"),
        default_window="[1,13]", default_kl_variant="Detailed"
    )})
    rows.append({"panel": "Panel B: Haifa competition (Aggregate)", **display_row(
        "",
        select_row(df, row_key="competition_aggregate", lp_family="Conventional DiD", lp_spec="Baseline",
                   kl_spec="Baseline", eta_interpretation_class="aggregate_regression_fallback"),
        default_window="[1,13]", default_kl_variant="Detailed"
    )})
    rows.append({"panel": "Panel B: Haifa competition (Aggregate)", **display_row(
        "",
        select_row(df, row_key="competition_aggregate", lp_family="Conventional DiD", lp_spec="+Tr",
                   kl_spec="Controls+Trend", eta_interpretation_class="aggregate_regression_fallback"),
        default_window="[1,13]", default_kl_variant="Detailed"
    )})
    rows.append({"panel": "Panel B: Haifa competition (Aggregate)", **display_row(
        "", None, default_window="[1,13]", default_lp_family="--", default_lp_spec="--",
        default_kl_spec="--", default_kl_variant="Detailed", default_eta_input="--"
    )})
    rows.append({"panel": "Panel B: Haifa competition (Aggregate)", **display_row(
        "", None, default_window="[1,13]", default_lp_family="--", default_lp_spec="--",
        default_kl_spec="--", default_kl_variant="Detailed", default_eta_input="--"
    )})

    # Panel C: Privatization - Legacy
    rows.append({"panel": "Panel C: Haifa privatization (Legacy)", **display_row(
        "Haifa privatization - Legacy",
        select_row(df, row_key="privatization_legacy", lp_family="NYT", lp_spec="Baseline",
                   kl_spec="Baseline", eta_interpretation_class="preferred_manual"),
        default_window="[1,7]", default_kl_variant="Compact"
    )})
    rows.append({"panel": "Panel C: Haifa privatization (Legacy)", **display_row(
        "",
        select_row(df, row_key="privatization_legacy", lp_family="NYT", lp_spec="+Tr",
                   kl_spec="Controls+Trend", eta_interpretation_class="preferred_manual"),
        default_window="[1,7]", default_kl_variant="Compact"
    )})
    rows.append({"panel": "Panel C: Haifa privatization (Legacy)", **display_row(
        "",
        select_row(df, row_key="privatization_legacy", lp_family="Conventional DiD", lp_spec="Baseline",
                   kl_spec="Baseline", eta_interpretation_class="preferred_manual"),
        default_window="[1,7]", default_kl_variant="Compact"
    )})
    rows.append({"panel": "Panel C: Haifa privatization (Legacy)", **display_row(
        "",
        select_row(df, row_key="privatization_legacy", lp_family="Conventional DiD", lp_spec="+Tr",
                   kl_spec="Controls+Trend", eta_interpretation_class="preferred_manual"),
        default_window="[1,7]", default_kl_variant="Compact"
    )})
    rows.append({"panel": "Panel C: Haifa privatization (Legacy)", **display_row(
        "",
        select_row(df, row_key="privatization_legacy", lp_family="NYT", lp_spec="Baseline",
                   kl_spec="Baseline", eta_interpretation_class="regression_robustness"),
        default_window="[1,7]", default_kl_variant="Compact"
    )})
    rows.append({"panel": "Panel C: Haifa privatization (Legacy)", **display_row(
        "",
        select_row(df, row_key="privatization_legacy", lp_family="Conventional DiD", lp_spec="+Tr",
                   kl_spec="Controls+Trend", eta_interpretation_class="regression_robustness"),
        default_window="[1,7]", default_kl_variant="Compact"
    )})

    # Panel D: Privatization - Aggregate
    rows.append({"panel": "Panel D: Haifa privatization (Aggregate)", **display_row(
        "Haifa privatization - Aggregate",
        select_row(df, row_key="privatization_aggregate", lp_family="NYT", lp_spec="Baseline",
                   kl_spec="Baseline", eta_interpretation_class="preferred_manual_aggregate"),
        default_window="[1,7]", default_kl_variant="Compact"
    )})
    rows.append({"panel": "Panel D: Haifa privatization (Aggregate)", **display_row(
        "",
        select_row(df, row_key="privatization_aggregate", lp_family="NYT", lp_spec="+Tr",
                   kl_spec="Controls+Trend", eta_interpretation_class="preferred_manual_aggregate"),
        default_window="[1,7]", default_kl_variant="Compact"
    )})
    rows.append({"panel": "Panel D: Haifa privatization (Aggregate)", **display_row(
        "",
        select_row(df, row_key="privatization_aggregate", lp_family="Conventional DiD", lp_spec="Baseline",
                   kl_spec="Baseline", eta_interpretation_class="preferred_manual_aggregate"),
        default_window="[1,7]", default_kl_variant="Compact"
    )})
    rows.append({"panel": "Panel D: Haifa privatization (Aggregate)", **display_row(
        "",
        select_row(df, row_key="privatization_aggregate", lp_family="Conventional DiD", lp_spec="+Tr",
                   kl_spec="Controls+Trend", eta_interpretation_class="preferred_manual_aggregate"),
        default_window="[1,7]", default_kl_variant="Compact"
    )})
    rows.append({"panel": "Panel D: Haifa privatization (Aggregate)", **display_row(
        "",
        select_row(df, row_key="privatization_aggregate", lp_family="NYT", lp_spec="Baseline",
                   kl_spec="Baseline", eta_interpretation_class="aggregate_regression_fallback"),
        default_window="[1,7]", default_kl_variant="Compact"
    )})
    rows.append({"panel": "Panel D: Haifa privatization (Aggregate)", **display_row(
        "",
        select_row(df, row_key="privatization_aggregate", lp_family="Conventional DiD", lp_spec="+Tr",
                   kl_spec="Controls+Trend", eta_interpretation_class="aggregate_regression_fallback"),
        default_window="[1,7]", default_kl_variant="Compact"
    )})

    return pd.DataFrame(rows)


def build_appendix_tex(df: pd.DataFrame, path: Path) -> None:
    # Expect 24 rows, 6 per panel.
    if len(df) != 24:
        raise ValueError(f"Expected 24 display rows for appendix template; got {len(df)}")

    def line_from_row(r: pd.Series, first_group_label: str | None = None) -> str:
        lead = first_group_label if first_group_label is not None else ""
        vals = [
            lead,
            r["post_window"],
            r["lp_family"],
            r["lp_spec"],
            r["kl_spec"],
            r["kl_variant"],
            r["eta_input"],
            r["TE"],
            r["dC"],
            r["eta"],
            r["CD"],
            r["share_explained"],
        ]
        return "        " + " & ".join(tex_escape(v) for v in vals) + r" \\"

    panels = [
        ("Panel A: Haifa competition (Legacy)", "Haifa competition (Legacy)"),
        ("Panel B: Haifa competition (Aggregate)", "Haifa competition (Aggregate)"),
        ("Panel C: Haifa privatization (Legacy)", "Haifa privatization (Legacy)"),
        ("Panel D: Haifa privatization (Aggregate)", "Haifa privatization (Aggregate)"),
    ]

    lines: List[str] = []
    lines.extend([
r"\begin{table}[!htbp]",
r"  \centering",
r"  \caption{Accounting decomposition of reform effects on $\ln(\mathrm{LP})$: full comparison across LP inputs, K/L inputs, and elasticity assumptions}",
r"  \label{tab:model2_full_results}",
r"  \begin{threeparttable}",
r"    \scriptsize",
r"    \setlength{\tabcolsep}{3.5pt}",
r"    \renewcommand{\arraystretch}{1.10}",
"",
r"    \begin{tabularx}{\textwidth}{@{}",
r"      >{\raggedright\arraybackslash}p{2.3cm}",
r"      >{\raggedright\arraybackslash}p{1.1cm}",
r"      >{\raggedright\arraybackslash}p{1.0cm}",
r"      >{\raggedright\arraybackslash}p{1.00cm}",
r"      >{\raggedright\arraybackslash}p{1.55cm}",
r"      >{\raggedright\arraybackslash}p{1.15cm}",
r"      >{\raggedright\arraybackslash}p{1.45cm}",
r"      *{5}{>{\centering\arraybackslash}X}",
r"    @{}}",
r"      \toprule",
r"      Reform type and impacted Haifa entity",
r"        & Post window",
r"        & LP family",
r"        & LP spec",
r"        & K/L spec",
r"        & K/L variant",
r"        & Elasticity input",
r"        & $TE^{r}$",
r"        & $\Delta C^{r}$",
r"        & $\hat{\eta}$",
r"        & $CD^{r}$",
r"        & $s^{r}$ \\",
r"      &",
r"        &",
r"        {\scriptsize(1A estimator)}",
r"        & {\scriptsize(1A shell)}",
r"        & {\scriptsize(1B shell)}",
r"        & {\scriptsize(1B grouping)}",
r"        & {\scriptsize(manual / regression)}",
r"        & {\scriptsize(total LP effect)}",
r"        & {\scriptsize(K/L effect)}",
r"        & {\scriptsize(elasticity)}",
r"        & {\scriptsize(capital-deepening-consistent component)}",
r"        & {\scriptsize(share explained)} \\",
r"      \midrule",
""
    ])

    cursor = 0
    for panel_title, group_label in panels:
        sub = df.iloc[cursor: cursor + 6].reset_index(drop=True)
        cursor += 6
        lines.append(f"      \\multicolumn{{12}}{{@{{}}c@{{}}}}{{\\textbf{{{tex_escape(panel_title)}}}}}\\\\")
        lines.append(r"      \addlinespace[2pt]")
        lines.append("")
        lines.append(f"      \\multirow[t]{{6}}{{3.35cm}}{{{tex_escape(group_label)}}}")
        # first row consumes lead from multirow line
        first = sub.iloc[0]
        vals = [
            first["post_window"], first["lp_family"], first["lp_spec"], first["kl_spec"],
            first["kl_variant"], first["eta_input"], first["TE"], first["dC"],
            first["eta"], first["CD"], first["share_explained"]
        ]
        lines[-1] += "\n" + "        & " + " & ".join(tex_escape(v) for v in vals) + r" \\"
        for j in range(1, 6):
            lines.append(line_from_row(sub.iloc[j], first_group_label=None))
        lines.append("")
        lines.append(r"      \addlinespace[0.8em]")

    # replace last addlinespace with bottomrule block
    if lines[-1] == r"      \addlinespace[0.8em]":
        lines = lines[:-1]

    lines.extend([
r"      \bottomrule",
r"    \end{tabularx}",
"",
r"    \begin{tablenotes}[flushleft]",
r"      \footnotesize",
r"      \setlength{\parindent}{0pt}",
r"      \item This appendix table reports the full supported set of Model 2 accounting decompositions in the fixed thesis layout. Windows are unchanged from the upstream Model 1A and Model 1B outputs. Cells are shown as `--' only when no true supporting result exists at that horizon under the current upstream design. “Elasticity input” reports both the source and the numerical value of $\hat{\eta}$.",
r"    \end{tablenotes}",
r"  \end{threeparttable}",
r"\end{table}",
""
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    thesis_root = find_thesis_root()
    parser = argparse.ArgumentParser(description="Format enhanced Model 2 accounting outputs into thesis-facing tables.")
    parser.add_argument("--main", type=Path, default=thesis_root / "Design" / "Output (new)" / "Model_2_final" / "Tables" / "model2_accounting_main.tsv")
    parser.add_argument("--appendix", type=Path, default=thesis_root / "Design" / "Output (new)" / "Model_2_final" / "Tables" / "model2_accounting_appendix.tsv")
    parser.add_argument("--diagnostics", type=Path, default=thesis_root / "Design" / "Output (new)" / "Model_2_final" / "Tables" / "model2_diagnostics_main.tsv")
    parser.add_argument("--diagnostics_all", type=Path, default=thesis_root / "Design" / "Output (new)" / "Model_2_final" / "Tables" / "model2_diagnostics_all.tsv")
    parser.add_argument("--outdir", type=Path, default=thesis_root / "Design" / "Output (new)" / "Model_2_final" / "Tables")
    parser.add_argument("--manifest", type=Path, default=thesis_root / "Design" / "Output (new)" / "Model_2_final" / "Tables" / "model2_accounting_manifest.json")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    require_manifest_ok(args.manifest)

    main_in = read_tsv(args.main)
    appendix_in = read_tsv(args.appendix)
    diag_in = read_tsv(args.diagnostics)
    diag_all_in = read_tsv(args.diagnostics_all)

    summary_df = build_summary_table(main_in)
    diag_df = build_diagnostics_table(diag_in)

    appendix_source = normalize_appendix_sources(appendix_in, diag_all_in)
    appendix_display_df = build_appendix_template_table(appendix_source)

    summary_tsv = args.outdir / "model2_table_summary.tsv"
    appendix_tsv = args.outdir / "model2_table_appendix.tsv"
    diag_tsv = args.outdir / "model2_table_diagnostics.tsv"
    summary_tex = args.outdir / "model2_table_summary.tex"
    appendix_tex = args.outdir / "model2_table_appendix.tex"
    diag_tex = args.outdir / "model2_table_diagnostics.tex"

    summary_df.to_csv(summary_tsv, sep="\t", index=False)
    appendix_display_df.to_csv(appendix_tsv, sep="\t", index=False)
    diag_df.to_csv(diag_tsv, sep="\t", index=False)

    write_simple_tex(summary_df, summary_tex,
                     "Accounting decomposition of reform effects on ln(LP): preferred specifications",
                     "tab:model2_summary")
    build_appendix_tex(appendix_display_df, appendix_tex)
    write_simple_tex(diag_df, diag_tex,
                     "Accounting decomposition diagnostics for preferred rows",
                     "tab:model2_diagnostics")

    print("=== Model_2_step4_to_tables_v13.py: done ===")
    print(f"Manifest        : {args.manifest}")
    print(f"Main input      : {args.main}")
    print(f"Appendix input  : {args.appendix}")
    print(f"Diagnostics in  : {args.diagnostics}")
    print(f"Diagnostics all : {args.diagnostics_all}")
    print(f"Summary TSV     : {summary_tsv}")
    print(f"Appendix TSV    : {appendix_tsv}")
    print(f"Diagnostics TSV : {diag_tsv}")
    print(f"Summary TeX     : {summary_tex}")
    print(f"Appendix TeX    : {appendix_tex}")
    print(f"Diagnostics TeX : {diag_tex}")


if __name__ == "__main__":
    main()
