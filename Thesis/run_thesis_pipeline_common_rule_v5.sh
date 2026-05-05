#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(pwd)"
[[ -d "Data" && -d "Design" ]] || {
  echo "ERROR: Run this from the THESIS root directory."
  exit 1
}

if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="Design/Output (new)/_pipeline_logs/$TS"
BACKUP_DIR="Design/Output (new)/_active_input_backups/$TS"
mkdir -p "$LOG_DIR" "$BACKUP_DIR"

MASTER_LOG="$LOG_DIR/run_all_pipeline_common_rule_v5.log"

echo "Pipeline run started at $(date)" | tee "$MASTER_LOG"
echo "Project root: $ROOT" | tee -a "$MASTER_LOG"
echo "Python bin: $(command -v "$PYTHON_BIN")" | tee -a "$MASTER_LOG"

run() {
  echo "" | tee -a "$MASTER_LOG"
  echo "============================================================" | tee -a "$MASTER_LOG"
  echo "RUNNING: $*" | tee -a "$MASTER_LOG"
  echo "============================================================" | tee -a "$MASTER_LOG"
  "$@" 2>&1 | tee -a "$MASTER_LOG"
}

require_file() {
  local f="$1"
  [[ -f "$f" ]] || {
    echo "ERROR: required file not found: $f" | tee -a "$MASTER_LOG"
    exit 1
  }
}

backup_and_copy() {
  local src="$1"
  local dest="$2"
  local tag="$3"

  require_file "$src"
  mkdir -p "$(dirname "$dest")"

  if [[ -f "$dest" ]]; then
    cp "$dest" "$BACKUP_DIR/${tag}__$(basename "$dest")"
  fi

  cp "$src" "$dest"
}

promote_flat_files() {
  local src_dir="$1"
  local dest_dir="$2"
  local tag="$3"

  mkdir -p "$dest_dir"
  shopt -s nullglob
  for src in "$src_dir"/*; do
    [[ -f "$src" ]] || continue
    backup_and_copy "$src" "$dest_dir/$(basename "$src")" "$tag"
  done
  shopt -u nullglob
}

run_optional_py() {
  local script="$1"
  shift || true
  if [[ -f "$script" ]]; then
    run "$PYTHON_BIN" "$script" "$@"
  else
    echo "SKIP (optional missing): $script" | tee -a "$MASTER_LOG"
  fi
}

###############################################################################
# CHECK REQUIRED NEW SCRIPTS
###############################################################################

require_file "Data/L_proxy/common_rule/Build_L_Proxy_CommonRule_v5.py"
require_file "Data/KL/Build_KL_Panel_new_v2.py"
require_file "Data/KL/Plot_KL_Series_v2.py"

###############################################################################
# 0) BUILD THE NEW LABOR PROXY (COMMON RULE V5)
###############################################################################

run "$PYTHON_BIN" "Data/L_proxy/common_rule/Build_L_Proxy_CommonRule_v5.py" \
  --tons "Data/Output/monthly_output_by_1000_tons_ports_and_terminals.tsv" \
  --teu "Data/Output/teu_monthly_plus_quarterly_by_port.tsv" \
  --kpis "Data/L_proxy/containers_kpis_annual_wide_filled.tsv" \
  --base-lproxy "Data/L_proxy/L_Proxy.tsv" \
  --out "Data/L_proxy/common_rule"

# Promote v5 labor proxy to canonical active locations so any downstream code
# that still expects Data/L_proxy/L_Proxy.tsv will now use the new proxy.
backup_and_copy \
  "Data/L_proxy/common_rule/L_Proxy_commonrule_v5.tsv" \
  "Data/L_proxy/L_Proxy.tsv" \
  "L_PROXY"

backup_and_copy \
  "Data/L_proxy/common_rule/labor_hours_monthly_terminal_commonrule_v5.tsv" \
  "Data/L_proxy/labor_hours_monthly_terminal.tsv" \
  "L_PROXY"

backup_and_copy \
  "Data/L_proxy/common_rule/labor_hours_monthly_port_commonrule_v5.tsv" \
  "Data/L_proxy/labor_hours_monthly_port.tsv" \
  "L_PROXY"

run_optional_py \
  "Data/L_proxy/Plot_L_Proxy_By_Strategy.py" \
  --strategy common_rule \
  --lproxy "Data/L_proxy/common_rule/L_Proxy_commonrule_v5.tsv" \
  --outdir "Data/L_proxy/common_rule/Visualizations"

###############################################################################
# 1) REBUILD LP USING THE NEW LABOR PROXY
###############################################################################

# S1 and S2 are labor-independent and can continue writing to canonical Data/LP.
run "$PYTHON_BIN" "Data/LP/Build_LP_Panel_S1_Tons.py" \
  --tons "Data/Output/monthly_output_by_1000_tons_ports_and_terminals.tsv" \
  --out "Data/LP"

run "$PYTHON_BIN" "Data/LP/Build_LP_Panel_S2_TEU.py" \
  --teu "Data/Output/teu_monthly_plus_quarterly_by_port.tsv" \
  --out "Data/LP"

# S3-S5 are written into a dedicated v5 folder first.
run "$PYTHON_BIN" "Data/LP/Build_LP_Panel_S3_LProxy.py" \
  --lproxy "Data/L_proxy/common_rule/L_Proxy_commonrule_v5.tsv" \
  --s2_term_quarter "Data/LP/S2_terminal_quarter_teu.tsv" \
  --out "Data/LP/common_rule_v5"

run "$PYTHON_BIN" "Data/LP/Build_LP_Panel_S4.py" \
  --s1_port_month_tons "Data/LP/S1_port_month_tons.tsv" \
  --s1_port_quarter_tons "Data/LP/S1_port_quarter_tons.tsv" \
  --s2_port_month_teu "Data/LP/S2_port_month_teu.tsv" \
  --s2_term_quarter_teu "Data/LP/S2_terminal_quarter_teu.tsv" \
  --s2_port_quarter_teu "Data/LP/S2_port_quarter_teu.tsv" \
  --s3_lproxy_clean "Data/LP/common_rule_v5/S3_lproxy_clean.tsv" \
  --s3_port_month_labor "Data/LP/common_rule_v5/S3_port_month_labor.tsv" \
  --s3_term_year_pi "Data/LP/common_rule_v5/S3_terminal_year_pi.tsv" \
  --out "Data/LP/common_rule_v5"

run "$PYTHON_BIN" "Data/LP/Build_LP_Panel_S5_Stack.py" \
  --haifa_m "Data/LP/common_rule_v5/LP_Haifa_port_month.tsv" \
  --ashdod_m "Data/LP/common_rule_v5/LP_Ashdod_port_month.tsv" \
  --haifa_legacy_q "Data/LP/common_rule_v5/LP_Haifa_Legacy_quarter.tsv" \
  --haifa_sipg_q "Data/LP/common_rule_v5/LP_Haifa_SIPG_quarter.tsv" \
  --ashdod_legacy_q "Data/LP/common_rule_v5/LP_Ashdod_Legacy_quarter.tsv" \
  --ashdod_hct_q "Data/LP/common_rule_v5/LP_Ashdod_HCT_quarter.tsv" \
  --out_dir "Data/LP/common_rule_v5"

# Promote the new LP outputs into canonical Data/LP so the models use the new LP.
promote_flat_files "Data/LP/common_rule_v5" "Data/LP" "LP_COMMON_RULE_V5"

###############################################################################
# 2) REBUILD K INTERPOLATION
###############################################################################

run "$PYTHON_BIN" "Data/K/interpolation_00_prepare_working_inputs_v8_hpc_operator_bridge_ready.py"
run "$PYTHON_BIN" "Data/K/interpolation_01_build_monthly_engine_v8_hpc_operator_linear_backloaded.py"
run "$PYTHON_BIN" "Data/K/interpolation_02_finalize_outputs_and_qc_v9_hpc_operator_linear_backloaded.py"

###############################################################################
# 3) REBUILD K/L USING THE NEW LABOR PROXY
###############################################################################

run "$PYTHON_BIN" "Data/KL/Build_KL_Panel_new_v2.py" \
  --lproxy "Data/L_proxy/common_rule/L_Proxy_commonrule_v5.tsv" \
  --k-long "Data/K/Interpolation Output/interpolation_02_monthly_entity_series_long.tsv" \
  --out-dir "Data/KL/common_rule_v5"

run "$PYTHON_BIN" "Data/KL/Plot_KL_Series_v2.py" \
  --lproxy "Data/L_proxy/common_rule/L_Proxy_commonrule_v5.tsv" \
  --k-wide "Data/K/Interpolation Output/interpolation_02_monthly_entity_series_wide.tsv" \
  --out-dir "Data/KL/common_rule_v5"

# Promote the new K/L outputs into canonical Data/KL so Model 1B and Model 2
# use the v5-based K/L panel.
promote_flat_files "Data/KL/common_rule_v5" "Data/KL" "KL_COMMON_RULE_V5"

if [[ -d "Data/KL/common_rule_v5/Visualization" ]]; then
  mkdir -p "Data/KL/Visualization"
  shopt -s nullglob
  for src in "Data/KL/common_rule_v5/Visualization"/*; do
    [[ -f "$src" ]] || continue
    backup_and_copy "$src" "Data/KL/Visualization/$(basename "$src")" "KL_VIS_COMMON_RULE_V5"
  done
  shopt -u nullglob
fi

# Some downstream workflows may still expect these older filenames.
backup_and_copy "Data/KL/KL_Panel_monthly.tsv" "Data/KL/KL_Panel_monthly_model2.tsv" "KL_ALIASES"
backup_and_copy "Data/KL/KL_Panel_monthly.tsv" "Data/KL/KL_Panel_monthly_diagnostic.tsv" "KL_ALIASES"

###############################################################################
# 4) OPTIONAL VISUALS / PREPROCESS
###############################################################################

run_optional_py "Design/Code (new)/preprocess_LP_Panel.py"

if [[ -f "Design/Output (new)/LP/Visuals/Plot_LP_Series.py" ]]; then
  run "$PYTHON_BIN" "Design/Output (new)/LP/Visuals/Plot_LP_Series.py"
elif [[ -f "Design/Code (new)/Plot_LP_Series.py" ]]; then
  run "$PYTHON_BIN" "Design/Code (new)/Plot_LP_Series.py"
else
  echo "SKIP (optional missing): LP visuals script" | tee -a "$MASTER_LOG"
fi

###############################################################################
# 5) MODEL 1A
###############################################################################

run "$PYTHON_BIN" "Design/Code (new)/Model_1A(v4).py"
run "$PYTHON_BIN" "Design/Code (new)/Model_1A_to_tables(v4).py"
run_optional_py "Design/Code (new)/Plot_Model_1A_event_study(v4).py"

###############################################################################
# 6) MODEL 1B
###############################################################################

run "$PYTHON_BIN" "Design/Code (new)/Model_1B(v4)_fixed2.py"
run "$PYTHON_BIN" "Design/Code (new)/Model_1B_relaxed(v4)_fixed.py"
run "$PYTHON_BIN" "Design/Code (new)/Model_1B_to_tables(v4)_fixed2.py"

if [[ -f "Design/Code (new)/Plot_Model_1B_event_study(v5).py" ]]; then
  run "$PYTHON_BIN" "Design/Code (new)/Plot_Model_1B_event_study(v5).py"
elif [[ -f "Design/Code (new)/Plot_Model_1B_event_study(v4).py" ]]; then
  run "$PYTHON_BIN" "Design/Code (new)/Plot_Model_1B_event_study(v4).py"
else
  echo "SKIP (optional missing): Model 1B plot script" | tee -a "$MASTER_LOG"
fi

###############################################################################
# 7) MODEL 2
###############################################################################

run "$PYTHON_BIN" "Design/Code (new)/Model_2_step1_build_panels.py"
run "$PYTHON_BIN" "Design/Code (new)/Model_2_step2_elasticity.py"
run "$PYTHON_BIN" "Design/Code (new)/Model_2_step3_accounting.py"
run "$PYTHON_BIN" "Design/Code (new)/Model_2_step4_to_tables.py"

echo "" | tee -a "$MASTER_LOG"
echo "DONE: full pipeline completed successfully at $(date)" | tee -a "$MASTER_LOG"
echo "Log written to: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "Backups written to: $BACKUP_DIR" | tee -a "$MASTER_LOG"
