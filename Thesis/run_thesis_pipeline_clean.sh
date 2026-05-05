#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Thesis pipeline: cleaned current version
###############################################################################

ROOT="/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis"
PY="$ROOT/.venv/bin/python"

cd "$ROOT" || exit 1

###############################################################################
# Config
###############################################################################

PIPELINE_TAG="common_rule_v5"

LPROXY_DIR="Data/L_proxy"
LPROXY_STRATEGY_DIR="$LPROXY_DIR/common_rule"

LP_DIR="Data/LP"
LP_BUILD_DIR="$LP_DIR/raw_from_l_v2"

K_DIR="Data/K"
K_OUT_DIR="$K_DIR/Interpolation Output"

KL_DIR="Data/KL"
KL_BUILD_DIR="$KL_DIR/$PIPELINE_TAG"

DESIGN_DIR="Design/Code (new)"

###############################################################################
# Helpers
###############################################################################

stage() {
  echo
  echo "###############################################################################"
  echo "# $1"
  echo "###############################################################################"
}

require_file() {
  [ -f "$1" ] || { echo "Missing required file: $1" >&2; exit 1; }
}

run_py() {
  echo "RUN: $1"
  "$PY" "$1" "${@:2}"
}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  [ -f "$src" ] && cp "$src" "$dst"
}

promote_files() {
  local src_dir="$1"
  local dst_dir="$2"
  shift 2
  for f in "$@"; do
    copy_if_exists "$src_dir/$f" "$dst_dir/$f"
  done
}

resolve_model_1a() {
  if [ -f "$DESIGN_DIR/Model_1A_v5_quarterly.py" ]; then
    echo "$DESIGN_DIR/Model_1A_v5_quarterly.py"
    return 0
  fi

  if [ -f "$DESIGN_DIR/Model_1A_v5_quarterly (2).py" ]; then
    echo "$DESIGN_DIR/Model_1A_v5_quarterly (2).py"
    return 0
  fi

  echo "Could not find updated quarterly Model 1A file." >&2
  exit 1
}

###############################################################################
# Resolve active updated Model 1A
###############################################################################

M1A="$(resolve_model_1a)"

###############################################################################
# Required files
###############################################################################

stage "0) Check required files"

require_file "$LPROXY_STRATEGY_DIR/Build_L_Proxy_CommonRule_v5.py"
require_file "$LP_DIR/Build_LP_Panel_S1_Tons_v2.py"
require_file "$LP_DIR/Build_LP_Panel_S2_TEU.py"
require_file "$LP_DIR/Build_LP_Panel_S3_LProxy.py"
require_file "$LP_DIR/Build_LP_Panel_raw_from_L_v2.py"
require_file "$KL_DIR/Build_KL_Panel_new_v2.py"

require_file "$K_DIR/interpolation_00_prepare_working_inputs_v8_hpc_operator_bridge_ready.py"
require_file "$K_DIR/interpolation_01_build_monthly_engine_v8_hpc_operator_linear_backloaded.py"
require_file "$K_DIR/interpolation_02_finalize_outputs_and_qc_v9_hpc_operator_linear_backloaded.py"

require_file "$DESIGN_DIR/Model_1B(v4)_fixed2.py"
require_file "$DESIGN_DIR/Model_1B_relaxed(v4)_fixed.py"
require_file "$DESIGN_DIR/Model_1B_to_tables(v4)_fixed2.py"
require_file "$DESIGN_DIR/Model_2_step1_build_panels.py"
require_file "$DESIGN_DIR/Model_2_step2_elasticity.py"
require_file "$DESIGN_DIR/Model_2_step3_accounting.py"
require_file "$DESIGN_DIR/Model_2_step4_to_tables.py"

###############################################################################
# 1) Labor proxy
###############################################################################

stage "1) Build labor proxy"

run_py "$LPROXY_STRATEGY_DIR/Build_L_Proxy_CommonRule_v5.py" \
  --tons "Data/Output/monthly_output_by_1000_tons_ports_and_terminals.tsv" \
  --teu "Data/Output/teu_monthly_plus_quarterly_by_port.tsv" \
  --kpis "Data/L_proxy/containers_kpis_annual_wide_filled.tsv" \
  --base-lproxy "Data/L_proxy/L_Proxy.tsv" \
  --out "$LPROXY_STRATEGY_DIR"

stage "1b) Promote labor proxy to canonical paths"

copy_if_exists \
  "$LPROXY_STRATEGY_DIR/L_Proxy_commonrule_v5.tsv" \
  "$LPROXY_DIR/L_Proxy.tsv"

copy_if_exists \
  "$LPROXY_STRATEGY_DIR/labor_hours_monthly_terminal_commonrule_v5.tsv" \
  "$LPROXY_DIR/labor_hours_monthly_terminal.tsv"

copy_if_exists \
  "$LPROXY_STRATEGY_DIR/labor_hours_monthly_port_commonrule_v5.tsv" \
  "$LPROXY_DIR/labor_hours_monthly_port.tsv"

###############################################################################
# 2) LP build
###############################################################################

stage "2) Build LP branch: raw_from_l_v2"

run_py "$LP_DIR/Build_LP_Panel_S1_Tons_v2.py" \
  --tons "Data/Output/monthly_output_by_1000_tons_ports_and_terminals.tsv" \
  --out "$LP_BUILD_DIR"

run_py "$LP_DIR/Build_LP_Panel_S2_TEU.py" \
  --teu "Data/Output/teu_monthly_plus_quarterly_by_port.tsv" \
  --out "$LP_BUILD_DIR"

run_py "$LP_DIR/Build_LP_Panel_S3_LProxy.py" \
  --lproxy "$LPROXY_DIR/L_Proxy.tsv" \
  --s2_term_quarter "$LP_BUILD_DIR/S2_terminal_quarter_teu.tsv" \
  --out "$LP_BUILD_DIR"

run_py "$LP_DIR/Build_LP_Panel_raw_from_L_v2.py" \
  --s1_port_month_tons "$LP_BUILD_DIR/S1_port_month_tons.tsv" \
  --s1_terminal_quarter_tons "$LP_BUILD_DIR/S1_terminal_quarter_tons.tsv" \
  --s2_port_month_teu "$LP_BUILD_DIR/S2_port_month_teu.tsv" \
  --s2_term_quarter_teu "$LP_BUILD_DIR/S2_terminal_quarter_teu.tsv" \
  --s3_lproxy_clean "$LP_BUILD_DIR/S3_lproxy_clean.tsv" \
  --s3_port_month_labor "$LP_BUILD_DIR/S3_port_month_labor.tsv" \
  --out "$LP_BUILD_DIR"

stage "2b) Promote LP outputs to canonical Data/LP"

promote_files "$LP_BUILD_DIR" "$LP_DIR" \
  "LP_Panel.tsv" \
  "LP_Haifa_port_month.tsv" \
  "LP_Ashdod_port_month.tsv" \
  "LP_Haifa_Legacy_quarter.tsv" \
  "LP_Haifa_SIPG_quarter.tsv" \
  "LP_Ashdod_Legacy_quarter.tsv" \
  "LP_Ashdod_HCT_quarter.tsv"

###############################################################################
# 3) K build
###############################################################################

stage "3) Build K"

run_py "$K_DIR/interpolation_00_prepare_working_inputs_v8_hpc_operator_bridge_ready.py"
run_py "$K_DIR/interpolation_01_build_monthly_engine_v8_hpc_operator_linear_backloaded.py"
run_py "$K_DIR/interpolation_02_finalize_outputs_and_qc_v9_hpc_operator_linear_backloaded.py"

###############################################################################
# 4) K/L build
###############################################################################

stage "4) Build K/L"

run_py "$KL_DIR/Build_KL_Panel_new_v2.py" \
  --lproxy "$LPROXY_DIR/L_Proxy.tsv" \
  --k-long "$K_OUT_DIR/interpolation_02_monthly_entity_series_long.tsv" \
  --out-dir "$KL_BUILD_DIR"

stage "4b) Promote K/L outputs to canonical Data/KL"

promote_files "$KL_BUILD_DIR" "$KL_DIR" \
  "KL_Panel_monthly.tsv" \
  "KL_monthly_series_long.tsv" \
  "KL_monthly_series_wide.tsv" \
  "kl_build_manifest.json"

stage "4c) Write downstream aliases"

copy_if_exists \
  "$KL_DIR/KL_Panel_monthly.tsv" \
  "$KL_DIR/KL_Panel_monthly_model2.tsv"

copy_if_exists \
  "$KL_DIR/KL_Panel_monthly.tsv" \
  "$KL_DIR/KL_Panel_monthly_diagnostic.tsv"

###############################################################################
# 5) Optional visuals
###############################################################################

stage "5) Optional visuals"

if [ -f "$LPROXY_DIR/Plot_L_Proxy_By_Strategy.py" ]; then
  run_py "$LPROXY_DIR/Plot_L_Proxy_By_Strategy.py" \
    --strategy common_rule \
    --lproxy "$LPROXY_STRATEGY_DIR/L_Proxy_commonrule_v5.tsv" \
    --outdir "$LPROXY_STRATEGY_DIR/Visualizations"
fi

if [ -f "$KL_DIR/Plot_KL_Series_v2.py" ]; then
  run_py "$KL_DIR/Plot_KL_Series_v2.py" \
    --lproxy "$LPROXY_DIR/L_Proxy.tsv" \
    --k-wide "$K_OUT_DIR/interpolation_02_monthly_entity_series_wide.tsv" \
    --out-dir "$KL_BUILD_DIR"
fi

###############################################################################
# 6) Model 1A
###############################################################################

stage "6) Model 1A"

run_py "$M1A"

###############################################################################
# 7) Model 1B
###############################################################################

stage "7) Model 1B"

run_py "$DESIGN_DIR/Model_1B(v4)_fixed2.py"
run_py "$DESIGN_DIR/Model_1B_relaxed(v4)_fixed.py"
run_py "$DESIGN_DIR/Model_1B_to_tables(v4)_fixed2.py"

###############################################################################
# 8) Model 2
###############################################################################

stage "8) Model 2"

run_py "$DESIGN_DIR/Model_2_step1_build_panels.py"
run_py "$DESIGN_DIR/Model_2_step2_elasticity.py"
run_py "$DESIGN_DIR/Model_2_step3_accounting.py"
run_py "$DESIGN_DIR/Model_2_step4_to_tables.py"

###############################################################################
# Done
###############################################################################

echo
echo "Updated full relevant pipeline completed successfully."
