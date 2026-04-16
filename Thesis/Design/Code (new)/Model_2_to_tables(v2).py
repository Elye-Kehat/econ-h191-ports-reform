"""
Model_2_to_tables(v2).py

Purpose
-------
Estimates Model 2 elasticities and writes table-ready outputs.

This script uses the analysis panels produced by build_model2_panels.py:

  - Design/Output (new)/Model_2A/model2a_terminal_panel.tsv
  - Design/Output (new)/Model_2B/model2b_cluster_panel.tsv

It then estimates (for each depreciation delta in {4%,6%,8%}):

A) Terminal elasticities (Table: Haifa terminals)
   - Haifa--Legacy:   log_LP ~ log_KL + t_index
   - Haifa--Bayport:  log_LP ~ log_KL + t_index
   - Pooled terminals: log_LP ~ log_KL + t_index + C(terminal)

B) Haifa port-cluster elasticity (optional robustness)
   - log_LP ~ log_KL + t_index

All regressions use heteroskedasticity-robust (HC1) standard errors.

Outputs
-------
Design/Output (new)/Model_2/Tables/
  - model2_elasticity_results.tsv
  - model2_cluster_elasticity_results.tsv
  - table_haifa_terminal_elasticity.tex
  - table_haifa_cluster_elasticity.tex

v2 tweaks
---------
1) Adds canonical columns for downstream scripts:
     eta    = coef_log_KL
     eta_se = se_log_KL
     pvalue = p_log_KL
   (Old columns are retained for backwards compatibility.)

2) Normalizes delta typing to avoid float-equality issues:
     delta is cast to numeric and rounded to 2 decimals.

3) Adds optional machine-friendly 'spec_key' alongside 'spec' (keeps 'spec'
   unchanged for LaTeX / readability).

Usage
-----
python Model_2_to_tables(v2).py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


###############################################################################
# Helpers
###############################################################################

def find_thesis_root(start: Optional[Path] = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        if (p / "Data").exists() and (p / "Design").exists():
            return p
    raise FileNotFoundError("Could not locate thesis root (expected folders: Data/, Design/).")


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "^{***}"
    if p < 0.05:
        return "^{**}"
    if p < 0.10:
        return "^{*}"
    return ""


def fmt(x: float, nd: int = 2) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{x:.{nd}f}"


def run_ols(df: pd.DataFrame, formula: str):
    """
    OLS with HC1 robust standard errors.
    """
    model = smf.ols(formula=formula, data=df)
    res = model.fit(cov_type="HC1")
    return res


def extract_logkl(res):
    """
    Extract coefficient, SE, p-value, N, R^2 for log_KL.
    """
    b = res.params.get("log_KL", np.nan)
    se = res.bse.get("log_KL", np.nan)
    p = res.pvalues.get("log_KL", np.nan)
    n = int(round(res.nobs))
    r2 = float(res.rsquared)
    return b, se, p, n, r2


###############################################################################
# Main estimation
###############################################################################

def main() -> None:
    print("=== Model_2_to_tables(v2): starting ===")
    THESIS_ROOT = find_thesis_root()
    print("THESIS_ROOT:", THESIS_ROOT)

    panel2a = THESIS_ROOT / "Design" / "Output (new)" / "Model_2A" / "model2a_terminal_panel.tsv"
    panel2b = THESIS_ROOT / "Design" / "Output (new)" / "Model_2B" / "model2b_cluster_panel.tsv"

    if not panel2a.exists() or not panel2b.exists():
        raise FileNotFoundError(
            "Missing Model 2 panels. Run build_model2_panels.py first.\n"
            f"Expected:\n  {panel2a}\n  {panel2b}"
        )

    df_term = pd.read_csv(panel2a, sep="\t")
    df_clus = pd.read_csv(panel2b, sep="\t")

    # Clean / coerce numeric
    for d in (df_term, df_clus):
        d["log_LP"] = pd.to_numeric(d["log_LP"], errors="coerce")
        d["log_KL"] = pd.to_numeric(d["log_KL"], errors="coerce")
        d["t_index"] = pd.to_numeric(d["t_index"], errors="coerce")
        if "delta" in d.columns:
            d["delta"] = pd.to_numeric(d["delta"], errors="coerce").round(2)

    df_term = df_term.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_LP", "log_KL", "t_index", "delta"])
    df_clus = df_clus.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_LP", "log_KL", "t_index", "delta"])

    DELTAS = [0.04, 0.06, 0.08]

    rows: List[Dict] = []

    # --- Terminal-specific
    for delta in DELTAS:
        for terminal in ["Haifa--Legacy", "Haifa--Bayport"]:
            sub = df_term[(df_term["terminal"] == terminal) & (df_term["delta"] == delta)].copy()
            sub = sub.dropna(subset=["log_LP", "log_KL", "t_index"])
            if len(sub) < 10:
                print(f"[WARN] Small sample for {terminal}, delta={delta:.2f}: n={len(sub)}")

            res = run_ols(sub, "log_LP ~ log_KL + t_index")
            b, se, p, n, r2 = extract_logkl(res)
            rows.append(
                {
                    "entity": terminal,
                    "delta": float(delta),
                    "spec": "TS+trend",
                    "spec_key": "preferred_single",
                    "coef_log_KL": b,
                    "se_log_KL": se,
                    "p_log_KL": p,
                    # canonical names (downstream-friendly)
                    "eta": b,
                    "eta_se": se,
                    "pvalue": p,
                    "N": n,
                    "R2": r2,
                }
            )

        # --- Pooled terminals (panel with terminal FE)
        sub = df_term[df_term["delta"] == delta].copy()
        res = run_ols(sub, "log_LP ~ log_KL + t_index + C(terminal)")
        b, se, p, n, r2 = extract_logkl(res)
        rows.append(
            {
                "entity": "Pooled Haifa terminals",
                "delta": float(delta),
                "spec": "Pooled+trend+FE",
                "spec_key": "preferred_pooled",
                "coef_log_KL": b,
                "se_log_KL": se,
                "p_log_KL": p,
                # canonical names
                "eta": b,
                "eta_se": se,
                "pvalue": p,
                "N": n,
                "R2": r2,
            }
        )

    out = pd.DataFrame(rows)

    # --- Cluster elasticity (robustness)
    rows_c: List[Dict] = []
    for delta in DELTAS:
        sub = df_clus[df_clus["delta"] == delta].copy()
        res = run_ols(sub, "log_LP ~ log_KL + t_index")
        b, se, p, n, r2 = extract_logkl(res)
        rows_c.append(
            {
                "entity": "Haifa port cluster",
                "delta": float(delta),
                "spec": "TS+trend",
                "spec_key": "preferred_cluster",
                "coef_log_KL": b,
                "se_log_KL": se,
                "p_log_KL": p,
                # canonical names
                "eta": b,
                "eta_se": se,
                "pvalue": p,
                "N": n,
                "R2": r2,
            }
        )
    out_c = pd.DataFrame(rows_c)

    # Write outputs
    tables_dir = THESIS_ROOT / "Design" / "Output (new)" / "Model_2" / "Tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    out_path = tables_dir / "model2_elasticity_results.tsv"
    out.to_csv(out_path, sep="\t", index=False)

    outc_path = tables_dir / "model2_cluster_elasticity_results.tsv"
    out_c.to_csv(outc_path, sep="\t", index=False)

    # Build LaTeX table (Haifa terminals, 9 columns)
    def cell(entity: str, delta: float) -> Tuple[str, str, str, str]:
        """
        Return (coef_str, se_str, N_str, R2_str) for entity×delta.
        """
        if entity == "Pooled":
            key = "Pooled Haifa terminals"
        elif entity == "Legacy":
            key = "Haifa--Legacy"
        elif entity == "Bayport":
            key = "Haifa--Bayport"
        else:
            key = entity

        r = out[(out["entity"] == key) & (out["delta"] == float(delta))]
        if r.empty:
            return "", "", "", ""
        r = r.iloc[0]
        coef = fmt(r["coef_log_KL"], 2) + stars(r["p_log_KL"])
        se = f"({fmt(r['se_log_KL'], 2)})"
        n = str(int(r["N"]))
        r2 = fmt(r["R2"], 3)
        return coef, se, n, r2

    # Assemble columns in requested order: Legacy (4/6/8), Bayport (4/6/8), Pooled (4/6/8)
    cols = []
    for ent in ["Legacy", "Bayport", "Pooled"]:
        for d in DELTAS:
            cols.append(cell(ent, d))

    # Row helpers
    coef_row = " & ".join([c[0] for c in cols])
    se_row = " & ".join([c[1] for c in cols])
    n_row = " & ".join([c[2] for c in cols])
    r2_row = " & ".join([c[3] for c in cols])

    tex_terminal = r"""
\begin{table}[t]
  \centering
  \caption{Haifa terminals: elasticity of $\ln(\mathrm{LP})$ with respect to $\ln(K/L)$ under alternative depreciation rates}
  \label{tab:haifa_terminal_elasticity}
  \begin{threeparttable}
    \small
    \setlength{\tabcolsep}{4pt}
    \renewcommand{\arraystretch}{1.1}
    \begin{tabular}{l*{9}{c}}
      \toprule
      & \multicolumn{3}{c}{Haifa--Legacy}
      & \multicolumn{3}{c}{Haifa--Bayport}
      & \multicolumn{3}{c}{Pooled Haifa terminals} \\
      \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
      & (1) $\delta=4\%$ & (2) $\delta=6\%$ & (3) $\delta=8\%$
      & (4) $\delta=4\%$ & (5) $\delta=6\%$ & (6) $\delta=8\%$
      & (7) $\delta=4\%$ & (8) $\delta=6\%$ & (9) $\delta=8\%$ \\
      \midrule
      $\ln(K/L)$ coefficient & {COEF_ROW} \\
                             & {SE_ROW} \\
      Observations           & {N_ROW} \\
      $R^2$                  & {R2_ROW} \\
      \bottomrule
    \end{tabular}
    \begin{tablenotes}[flushleft]
      \footnotesize
      \item Notes: Each column reports $\hat{\eta}$ from a regression of $\ln(\mathrm{LP})$ on $\ln(K/L)$ at monthly frequency.
      Terminal-specific specifications are time-series regressions with a linear time trend ($t$).
      The pooled specification stacks Haifa--Legacy and Haifa--Bayport and includes terminal fixed effects and a common linear time trend.
      Heteroskedasticity-robust (HC1) standard errors are shown in parentheses.
    \end{tablenotes}
  \end{threeparttable}
\end{table}
""".strip("\n").replace("{COEF_ROW}", coef_row).replace("{SE_ROW}", se_row).replace("{N_ROW}", n_row).replace("{R2_ROW}", r2_row)

    tex_path = tables_dir / "table_haifa_terminal_elasticity.tex"
    tex_path.write_text(tex_terminal)

    # LaTeX table for cluster elasticity (3 columns)
    def cell_c(delta: float) -> Tuple[str, str, str, str]:
        r = out_c[out_c["delta"] == float(delta)].iloc[0]
        coef = fmt(r["coef_log_KL"], 2) + stars(r["p_log_KL"])
        se = f"({fmt(r['se_log_KL'], 2)})"
        n = str(int(r["N"]))
        r2 = fmt(r["R2"], 3)
        return coef, se, n, r2

    ccols = [cell_c(d) for d in DELTAS]
    c_coef = " & ".join([c[0] for c in ccols])
    c_se = " & ".join([c[1] for c in ccols])
    c_n = " & ".join([c[2] for c in ccols])
    c_r2 = " & ".join([c[3] for c in ccols])

    tex_cluster = r"""
\begin{table}[t]
  \centering
  \caption{Haifa port cluster: elasticity of $\ln(\mathrm{LP})$ with respect to $\ln(K/L)$ under alternative depreciation rates}
  \label{tab:haifa_cluster_elasticity}
  \begin{threeparttable}
    \small
    \setlength{\tabcolsep}{8pt}
    \renewcommand{\arraystretch}{1.1}
    \begin{tabular}{l*{3}{c}}
      \toprule
      & (1) $\delta=4\%$ & (2) $\delta=6\%$ & (3) $\delta=8\%$ \\
      \midrule
      $\ln(K/L)$ coefficient & {COEF_ROW} \\
                             & {SE_ROW} \\
      Observations           & {N_ROW} \\
      $R^2$                  & {R2_ROW} \\
      \bottomrule
    \end{tabular}
    \begin{tablenotes}[flushleft]
      \footnotesize
      \item Notes: Entries report the coefficient on $\ln(K/L)_t$ from a monthly time-series regression of $\ln(\mathrm{LP})_t$
      on $\ln(K/L)_t$ for the Haifa port cluster, including a linear time trend.
      No treatment indicators or event-time dummies enter this equation.
      Heteroskedasticity-robust (HC1) standard errors are shown in parentheses.
    \end{tablenotes}
  \end{threeparttable}
\end{table}
""".strip("\n").replace("{COEF_ROW}", c_coef).replace("{SE_ROW}", c_se).replace("{N_ROW}", c_n).replace("{R2_ROW}", c_r2)

    texc_path = tables_dir / "table_haifa_cluster_elasticity.tex"
    texc_path.write_text(tex_cluster)

    print("Wrote:", out_path)
    print("Wrote:", outc_path)
    print("Wrote:", tex_path)
    print("Wrote:", texc_path)
    print("=== Model_2_to_tables(v2): done ===")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# DIAGNOSTIC / DESIGN NOTE (v2 outputs)
#
# This v2 script is primarily a schema-stability upgrade for downstream
# decomposition (Model_2_mediation):
#
# 1) Canonical output columns:
#    - Adds eta and eta_se as duplicates of coef_log_KL and se_log_KL
#    - Adds pvalue as a duplicate of p_log_KL
#    This avoids fragile "column renaming / normalization" logic downstream.
#
# 2) Delta robustness:
#    - Casts delta to numeric and rounds to 2 decimals before filtering.
#    This prevents float-equality mismatches (e.g., 0.06000000001 != 0.06).
#
# 3) Spec naming:
#    - Keeps human-readable spec strings used in tables:
#        * "TS+trend" for single-series terminal and cluster regressions
#        * "Pooled+trend+FE" for pooled-terminal regression with C(terminal)
#    - Adds machine-friendly spec_key (preferred_single / preferred_pooled /
#      preferred_cluster) for future-proof lookup.
#
# Expected output structure:
#   - model2_elasticity_results.tsv: 9 rows = 3 deltas × {Legacy,Bayport,Pooled}
#   - model2_cluster_elasticity_results.tsv: 3 rows = 3 deltas × {Cluster}
#   - LaTeX tables should contain no unreplaced placeholders.
#
# Substantive note:
#   - Elasticity magnitudes/signs may look odd under the current LP proxy and
#     imputed K/L series. This file is not responsible for identification;
#     it simply runs the specified time-series / pooled regressions with HC1.
#     Revisit interpretation once true monthly labor-hours are incorporated.
# ----------------------------------------------------------------------