#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(pwd)"
[[ -d "Data" && -d "Design" ]] || {
  echo "ERROR: Run this from the THESIS root directory."
  exit 1
}

PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="Design/Output (new)/_pipeline_logs/$TS"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/run_all_pipeline.log"

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

run_one_of_python() {
  for f in "$@"; do
    if [[ -f "$f" ]]; then
      run "$PYTHON_BIN" "$f"
      return 0
    fi
  done
  echo "ERROR: none of these candidate scripts exist:" | tee -a "$MASTER_LOG"
  for f in "$@"; do
    echo "  - $f" | tee -a "$MASTER_LOG"
  done
  exit 1
}

run_optional_one_of_python() {
  for f in "$@"; do
    if [[ -f "$f" ]]; then
      run "$PYTHON_BIN" "$f"
      return 0
    fi
  done
  echo "SKIP (optional missing): no candidate found." | tee -a "$MASTER_LOG"
}

###############################################################################
# 0) USE EXISTING LABOR PROXY
###############################################################################

if [[ ! -f "Data/L_proxy/L_Proxy.tsv" ]]; then
  if [[ -f "Data/L_proxy/common_rule/L_Proxy_commonrule_v3.tsv" ]]; then
    cp "Data/L_proxy/common_rule/L_Proxy_commonrule_v3.tsv" "Data/L_proxy/L_Proxy.tsv"
    echo "Copied common_rule labor proxy to Data/L_proxy/L_Proxy.tsv" | tee -a "$MASTER_LOG"
  else
    echo "ERROR: neither Data/L_proxy/L_Proxy.tsv nor Data/L_proxy/common_rule/L_Proxy_commonrule_v3.tsv exists." | tee -a "$MASTER_LOG"
    exit 1
  fi
else
  echo "Found Data/L_proxy/L_Proxy.tsv" | tee -a "$MASTER_LOG"
fi

###############################################################################
# 1) LP PANEL
###############################################################################

if [[ -f "Data/LP/Build_LP_Panel.py" ]]; then
  run "$PYTHON_BIN" "Data/LP/Build_LP_Panel.py" \
    --tons "Data/Output/monthly_output_by_1000_tons_ports_and_terminals.tsv" \
    --teu "Data/Output/teu_monthly_plus_quarterly_by_port.tsv" \
    --lproxy "Data/L_proxy/L_Proxy.tsv" \
    --out "Data/LP" \
    --cutover_month 2021-09 \
    --winsor_low 0.01 \
    --winsor_high 0.99
else
  run "$PYTHON_BIN" "Data/LP/Build_LP_Panel_S1_Tons.py" \
    --tons "Data/Output/monthly_output_by_1000_tons_ports_and_terminals.tsv" \
    --out "Data/LP"

  run "$PYTHON_BIN" "Data/LP/Build_LP_Panel_S2_TEU.py" \
    --teu "Data/Output/teu_monthly_plus_quarterly_by_port.tsv" \
    --out "Data/LP"

  run "$PYTHON_BIN" "Data/LP/Build_LP_Panel_S3_LProxy.py" \
    --lproxy "Data/L_proxy/L_Proxy.tsv" \
    --s2_term_quarter "Data/LP/S2_terminal_quarter_teu.tsv" \
    --out "Data/LP"

  run "$PYTHON_BIN" "Data/LP/Build_LP_Panel_S4.py" \
    --winsor_low 0.01 \
    --winsor_high 0.99 \
    --monthly_start 201801 \
    --monthly_end 202108 \
    --quarterly_start 2021Q3 \
    --quarterly_end 2024Q4 \
    --out "Data/LP"

  run "$PYTHON_BIN" "Data/LP/Build_LP_Panel_S5_Stack.py" \
    --out_dir "Data/LP"
fi

###############################################################################
# 2) K INTERPOLATION
###############################################################################

run "$PYTHON_BIN" "Data/K/interpolation_00_prepare_working_inputs_v8_hpc_operator_bridge_ready.py"
run "$PYTHON_BIN" "Data/K/interpolation_01_build_monthly_engine_v8_hpc_operator_linear_backloaded.py"
run "$PYTHON_BIN" "Data/K/interpolation_02_finalize_outputs_and_qc_v9_hpc_operator_linear_backloaded.py"

###############################################################################
# 3) K/L PANEL
###############################################################################

run_one_of_python \
  "Design/Code (new)/Build_KL_Panel.py" \
  "Design/Code (new)/build_KL_Panel.py"

###############################################################################
# 4) OPTIONAL LP PREPROCESSOR
###############################################################################

run_optional_one_of_python \
  "Design/Code (new)/preprocess_LP_Panel.py"

###############################################################################
# 5) MODEL 1A
###############################################################################

run_one_of_python "Design/Code (new)/Model_1A(v4).py"
run_one_of_python "Design/Code (new)/Model_1A_to_tables(v4).py"
run_optional_one_of_python "Design/Code (new)/Plot_Model_1A_event_study(v4).py"

###############################################################################
# 6) MODEL 1B
###############################################################################

run_one_of_python \
  "Design/Code (new)/Model_1B(v4)_fixed2.py" \
  "Design/Code (new)/Model_1B(v4).py"

run_one_of_python \
  "Design/Code (new)/Model_1B_relaxed(v4)_fixed.py" \
  "Design/Code (new)/Model_1B_relaxed(v4).py"

run_one_of_python \
  "Design/Code (new)/Model_1B_to_tables(v4)_fixed2.py" \
  "Design/Code (new)/Model_1B_to_tables(v4).py"

run_optional_one_of_python \
  "Design/Code (new)/Plot_Model_1B_event_study(v5).py" \
  "Design/Code (new)/Plot_Model_1B_event_study(v4).py"

###############################################################################
# 7) MODEL 2
###############################################################################

run_one_of_python "Design/Code (new)/Model_2_step1_build_panels.py"
run_one_of_python "Design/Code (new)/Model_2_step2_elasticity.py"
run_one_of_python "Design/Code (new)/Model_2_step3_accounting.py"
run_one_of_python "Design/Code (new)/Model_2_step4_to_tables.py"

###############################################################################
# 8) OPTIONAL VISUALS
###############################################################################

run_optional_one_of_python \
  "Data/L_proxy/Plot_L_Proxy_By_Strategy.py" \
  "Design/Output (new)/L_Proxy/Visualizations/Plot_L_Proxy_Series.py" \
  "Design/Code (new)/Plot_L_Proxy_Series.py"

run_optional_one_of_python \
  "Design/Output (new)/LP/Visuals/Plot_LP_Series.py" \
  "Design/Code (new)/Plot_LP_Series.py"

echo "" | tee -a "$MASTER_LOG"
echo "DONE: full pipeline completed successfully at $(date)" | tee -a "$MASTER_LOG"
echo "Log written to: $MASTER_LOG" | tee -a "$MASTER_LOG"
