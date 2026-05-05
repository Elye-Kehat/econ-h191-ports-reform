cat > run_thesis_pipeline_v6.sh <<'BASH'
set -euo pipefail

###############################################################################
# Thesis pipeline: v6 tons-only L proxy + v6 raw LP + Model 1A v8
###############################################################################

ROOT="/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis"
PY="$ROOT/.venv/bin/python"

cd "$ROOT" || exit 1

###############################################################################
# Config
###############################################################################

PIPELINE_TAG="common_rule_v6_tonsonly"

LPROXY_DIR="Data/L_proxy"
LPROXY_CODE_DIR="$LPROXY_DIR/common_rule"
LPROXY_STRATEGY_DIR="$LPROXY_DIR/common_rule_v6_tonsonly"

LP_DIR="Data/LP"
LP_BUILD_DIR="$LP_DIR/raw_from_l_v6_tonsonly"

K_DIR="Data/K"
K_OUT_DIR="$K_DIR/Interpolation Output"

KL_DIR="Data/KL"
KL_BUILD_DIR="$KL_DIR/$PIPELINE_TAG"

DESIGN_DIR="Design/Code (new)"

LPROXY_BUILDER="$LPROXY_CODE_DIR/Build_L_Proxy_CommonRule_v6_tonsonly.py"
LP_S1="$LP_DIR/Build_LP_Panel_S1_Tons_v2.py"
LP_S2="$LP_DIR/Build_LP_Panel_S2_TEU.py"
LP_RAW="$LP_DIR/Build_LP_Panel_raw_from_L_v6_tonsonly.py"

M1A="$DESIGN_DIR/Model_1A_v8.py"
M1A_LP_INPUT="Data/LP/raw_from_l_v6_tonsonly/LP_Panel.tsv"

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

###############################################################################
# 0) Check required files
###############################################################################

stage "0) Check required files"

require_file "$M1A"
require_file "$LPROXY_BUILDER"
require_file "$LP_S1"
require_file "$LP_S2"
require_file "$LP_RAW"
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

mkdir -p "$LPROXY_STRATEGY_DIR" "$LP_BUILD_DIR" "$KL_BUILD_DIR"

###############################################################################
# 1) Labor proxy: v6 tons-only within-year allocation
###############################################################################

stage "1) Build labor proxy: common_rule_v6_tonsonly"

run_py "$LPROXY_BUILDER" \
  --tons "Data/Output/monthly_output_by_1000_tons_ports_and_terminals.tsv" \
  --teu "Data/Output/teu_monthly_plus_quarterly_by_port.tsv" \
  --kpis "Data/L_proxy/containers_kpis_annual_wide_filled.tsv" \
  --base-lproxy "Data/L_proxy/L_Proxy.tsv" \
  --out "$LPROXY_STRATEGY_DIR" \
  --also-write-canonical

stage "1b) Promote labor side outputs to canonical helper paths"

copy_if_exists \
  "$LPROXY_STRATEGY_DIR/labor_hours_monthly_terminal_commonrule_v6_tonsonly.tsv" \
  "$LPROXY_DIR/labor_hours_monthly_terminal.tsv"

copy_if_exists \
  "$LPROXY_STRATEGY_DIR/labor_hours_monthly_port_commonrule_v6_tonsonly.tsv" \
  "$LPROXY_DIR/labor_hours_monthly_port.tsv"

###############################################################################
# 2) LP build: v6 raw throughput over new L, including quarterly port aggregates
###############################################################################

stage "2) Build LP branch: raw_from_l_v6_tonsonly"

run_py "$LP_S1" \
  --tons "Data/Output/monthly_output_by_1000_tons_ports_and_terminals.tsv" \
  --out "$LP_BUILD_DIR"

run_py "$LP_S2" \
  --teu "Data/Output/teu_monthly_plus_quarterly_by_port.tsv" \
  --out "$LP_BUILD_DIR"

run_py "$LP_RAW" \
  --s1_port_month_tons "$LP_BUILD_DIR/S1_port_month_tons.tsv" \
  --s1_terminal_quarter_tons "$LP_BUILD_DIR/S1_terminal_quarter_tons.tsv" \
  --s2_port_month_teu "$LP_BUILD_DIR/S2_port_month_teu.tsv" \
  --s2_term_quarter_teu "$LP_BUILD_DIR/S2_terminal_quarter_teu.tsv" \
  --lproxy "$LPROXY_STRATEGY_DIR/L_Proxy_commonrule_v6_tonsonly.tsv" \
  --out "$LP_BUILD_DIR" \
  --also-write-canonical

stage "2b) Promote LP component outputs to canonical Data/LP"

promote_files "$LP_BUILD_DIR" "$LP_DIR" \
  "LP_Haifa_port_month.tsv" \
  "LP_Ashdod_port_month.tsv" \
  "LP_Haifa_port_quarter.tsv" \
  "LP_Ashdod_port_quarter.tsv" \
  "LP_Haifa_Legacy_quarter.tsv" \
  "LP_Haifa_SIPG_quarter.tsv" \
  "LP_Ashdod_Legacy_quarter.tsv" \
  "LP_Ashdod_HCT_quarter.tsv" \
  "LP_Panel.tsv" \
  "LP_raw_qa.tsv"

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
# 5) Model 1A v8
###############################################################################

stage "5) Model 1A v8"

run_py "$M1A" \
  --lp "$M1A_LP_INPUT"

###############################################################################
# 6) Model 1B
###############################################################################

stage "6) Model 1B"

run_py "$DESIGN_DIR/Model_1B(v4)_fixed2.py"
run_py "$DESIGN_DIR/Model_1B_relaxed(v4)_fixed.py"
run_py "$DESIGN_DIR/Model_1B_to_tables(v4)_fixed2.py"

###############################################################################
# 7) Model 2
###############################################################################

stage "7) Model 2"

run_py "$DESIGN_DIR/Model_2_step1_build_panels.py"
run_py "$DESIGN_DIR/Model_2_step2_elasticity.py"
run_py "$DESIGN_DIR/Model_2_step3_accounting.py"
run_py "$DESIGN_DIR/Model_2_step4_to_tables.py"

###############################################################################
# Done
###############################################################################

echo
echo "Updated full relevant pipeline completed successfully."
BASH

bash run_thesis_pipeline_v6.sh