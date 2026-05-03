#!/usr/bin/env bash
set -Eeuo pipefail

# Run from the THESIS project root.
# Optional:
#   export PYTHON_BIN=python3
#   export RUN_L_BACKFILL=1

ROOT="$(pwd)"
[[ -d "Data" && -d "Design" ]] || {
  echo "ERROR: Run this from the THESIS root directory."
  exit 1
}

PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN not found on PATH."
  exit 1
fi

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
# 1) LABOR PROXY
###############################################################################

run "$PYTHON_BIN" "Data/L_proxy/construct_L.py"
run "$PYTHON_BIN" "Data/L_proxy/verify_delta.py"

if [[ "${RUN_L_BACKFILL:-0}" == "1" ]]; then
  if [[ -f "Data/L_proxy/build_labor_proxy_backfill" ]]; then
    run "Data/L_proxy/build_labor_proxy_backfill"
  elif [[ -f "Data/L_proxy/build_labor_proxy_backfill.py" ]]; then
    run "$PYTHON_BIN" "Data/L_proxy/build_labor_proxy_backfill.py"
  else
    echo "SKIP (optional missing): labor backfill script not found." | tee -a "$MASTER_LOG"
  fi
fi

run "$PYTHON_BIN" "Data/L_proxy/Join.py"

###############################################################################
# 2) LP PANEL
###############################################################################

if [[ -f "Data/LP/Build_LP_Panel.py" ]]; then
  run "$PYTHON_BIN" "Data/LP/Build_LP_Panel.py"
else
  run "$PYTHON_BIN" "Data/LP/Build_LP_Panel_S1_Tons.py"
  run "$PYTHON_BIN" "Data/LP/Build_LP_Panel_S2_TEU.py"
  run "$PYTHON_BIN" "Data/LP/Build_LP_Panel_S3_LProxy.py"
  run "$PYTHON_BIN" "Data/LP/Build_LP_Panel_S4.py"
  run "$PYTHON_BIN" "Data/LP/Build_LP_Panel_S5_Stack.py"
fi

###############################################################################
# 3) K INTERPOLATION
###############################################################################

run "$PYTHON_BIN" "Data/K/interpolation_00_prepare_working_inputs_v8_hpc_operator_bridge_ready.py"
run "$PYTHON_BIN" "Data/K/interpolation_01_build_monthly_engine_v8_hpc_operator_linear_backloaded.py"
run "$PYTHON_BIN" "Data/K/interpolation_02_finalize_outputs_and_qc_v9_hpc_operator_linear_backloaded.py"

###############################################################################
# 4) K/L PANEL
###############################################################################

run_one_of_python \
  "Design/Code (new)/Build_KL_Panel.py" \
  "Design/Code (new)/build_KL_Panel.py"

###############################################################################
# 5) OPTIONAL LP PREPROCESSOR
###############################################################################

run_optional_one_of_python \
  "Design/Code (new)/preprocess_LP_Panel.py"

###############################################################################
# 6) MODEL 1A
###############################################################################

run_one_of_python "Design/Code (new)/Model_1A(v4).py"
run_one_of_python "Design/Code (new)/Model_1A_to_tables(v4).py"
run_optional_one_of_python "Design/Code (new)/Plot_Model_1A_event_study(v4).py"

###############################################################################
# 7) MODEL 1B
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
# 8) MODEL 2
###############################################################################

run_one_of_python "Design/Code (new)/Model_2_step1_build_panels.py"
run_one_of_python "Design/Code (new)/Model_2_step2_elasticity.py"
run_one_of_python "Design/Code (new)/Model_2_step3_accounting.py"
run_one_of_python "Design/Code (new)/Model_2_step4_to_tables.py"

###############################################################################
# 9) OPTIONAL VISUALS
###############################################################################

run_optional_one_of_python \
  "Design/Output (new)/L_Proxy/Visualizations/Plot_L_Proxy_Series.py" \
  "Design/Code (new)/Plot_L_Proxy_Series.py"

run_optional_one_of_python \
  "Design/Output (new)/LP/Visuals/Plot_LP_Series.py" \
  "Design/Code (new)/Plot_LP_Series.py"

echo "" | tee -a "$MASTER_LOG"
echo "DONE: full pipeline completed successfully at $(date)" | tee -a "$MASTER_LOG"
echo "Log written to: $MASTER_LOG" | tee -a "$MASTER_LOG"
