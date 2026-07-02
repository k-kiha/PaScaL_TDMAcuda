#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR"}
RUN_TIMESTAMP=${TIMESTAMP:-$(date +%y%m%d_%H%M%S)}

STUDY_PRESET=${STUDY_PRESET:-portfolio}
ITERATIONS=${ITERATIONS:-10}
BASELINE_NP=${BASELINE_NP:-2}
SCALING_NP_LIST=${SCALING_NP_LIST:-"2 4 8"}
RUN_NP1_REFERENCE=${RUN_NP1_REFERENCE:-1}
CXX_DEFAULT_MPI_MODES=${CXX_DEFAULT_MPI_MODES:-device}
MPI_MODE_LIST=${MPI_MODE_LIST:-"device host"}
RUN_FORTRAN=${RUN_FORTRAN:-1}
RUN_CXX=${RUN_CXX:-1}
TDMA_THREADS=${TDMA_THREADS:-128}
REDUCED_THREADS=${REDUCED_THREADS:-128}
SKIP_DRY_RUN=${SKIP_DRY_RUN:-0}
DRY_RUN_ONLY=${DRY_RUN_ONLY:-0}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=${DEFAULT_CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
fi

mkdir -p "$OUTPUT_DIR"

TIMING_OUT="$OUTPUT_DIR/tdma_total_profile_${RUN_TIMESTAMP}.csv"
CORRECTNESS_OUT="$OUTPUT_DIR/tdma_correctness_${RUN_TIMESTAMP}.csv"
ENV_OUT="$OUTPUT_DIR/tdma_environment_${RUN_TIMESTAMP}.txt"
MANIFEST_OUT="$OUTPUT_DIR/tdma_case_manifest_${RUN_TIMESTAMP}.csv"
LOG_OUT="$OUTPUT_DIR/tdma_full_study_${RUN_TIMESTAMP}.log"

DRY_TIMESTAMP="${RUN_TIMESTAMP}_dryrun"
DRY_TIMING_OUT="$OUTPUT_DIR/tdma_total_profile_${DRY_TIMESTAMP}.csv"
DRY_CORRECTNESS_OUT="$OUTPUT_DIR/tdma_correctness_${DRY_TIMESTAMP}.csv"
DRY_ENV_OUT="$OUTPUT_DIR/tdma_environment_${DRY_TIMESTAMP}.txt"
DRY_MANIFEST_OUT="$OUTPUT_DIR/tdma_case_manifest_${DRY_TIMESTAMP}.csv"

usage() {
    cat <<'USAGE'
Usage:
  ./run_full_study.sh
  DRY_RUN_ONLY=1 ./run_full_study.sh
  SKIP_DRY_RUN=1 ./run_full_study.sh

Purpose:
  Run the full PaScaL_TDMAcuda portfolio Study in one command.

Defaults:
  STUDY_PRESET=portfolio
  BASELINE_NP=2
  SCALING_NP_LIST="2 4 8"
  ITERATIONS=10
  CXX_DEFAULT_MPI_MODES=device
  MPI_MODE_LIST="device host"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 if it is not already set

Useful overrides:
  STUDY_PRESET=quick ITERATIONS=3 ./run_full_study.sh
  OUTPUT_DIR=/path/to/results ./run_full_study.sh
  DEFAULT_CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_full_study.sh
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

check_executable() {
    local exe="$1"
    if [[ ! -x "$exe" ]]; then
        echo "error: missing executable: $exe" >&2
        echo "build first from PaScaL_TDMAcuda root: make all CUDA_ARCH=90" >&2
        exit 1
    fi
}

print_summary() {
    cat <<EOF
== PaScaL_TDMAcuda full Study ==
study_preset=$STUDY_PRESET
timestamp=$RUN_TIMESTAMP
output_dir=$OUTPUT_DIR
cuda_visible_devices=$CUDA_VISIBLE_DEVICES
baseline_np=$BASELINE_NP
scaling_np_list=$SCALING_NP_LIST
iterations=$ITERATIONS
cxx_default_mpi_modes=$CXX_DEFAULT_MPI_MODES
mpi_mode_list=$MPI_MODE_LIST
run_fortran=$RUN_FORTRAN
run_cxx=$RUN_CXX

outputs:
  $TIMING_OUT
  $CORRECTNESS_OUT
  $ENV_OUT
  $MANIFEST_OUT
  $LOG_OUT
EOF
}

run_sweep() {
    local dry_run="$1"
    local timestamp="$2"
    local timing_out="$3"
    local correctness_out="$4"
    local env_out="$5"
    local manifest_out="$6"

    STUDY_PRESET="$STUDY_PRESET" \
    ITERATIONS="$ITERATIONS" \
    BASELINE_NP="$BASELINE_NP" \
    SCALING_NP_LIST="$SCALING_NP_LIST" \
    RUN_NP1_REFERENCE="$RUN_NP1_REFERENCE" \
    CXX_DEFAULT_MPI_MODES="$CXX_DEFAULT_MPI_MODES" \
    MPI_MODE_LIST="$MPI_MODE_LIST" \
    RUN_FORTRAN="$RUN_FORTRAN" \
    RUN_CXX="$RUN_CXX" \
    TDMA_THREADS="$TDMA_THREADS" \
    REDUCED_THREADS="$REDUCED_THREADS" \
    TIMESTAMP="$timestamp" \
    OUT="$timing_out" \
    CORRECTNESS_OUT="$correctness_out" \
    ENV_OUT="$env_out" \
    MANIFEST_OUT="$manifest_out" \
    DRY_RUN="$dry_run" \
    "$SCRIPT_DIR/run_study_sweep.sh"
}

print_summary

if [[ "$DRY_RUN_ONLY" == "1" ]]; then
    echo
    echo "== Dry-run only =="
    run_sweep 1 "$DRY_TIMESTAMP" "$DRY_TIMING_OUT" "$DRY_CORRECTNESS_OUT" "$DRY_ENV_OUT" "$DRY_MANIFEST_OUT"
    echo
    echo "dry-run manifest: $DRY_MANIFEST_OUT"
    exit 0
fi

if [[ "$RUN_FORTRAN" == "1" ]]; then
    check_executable "$SCRIPT_DIR/example_fortran_profile"
fi
if [[ "$RUN_CXX" == "1" ]]; then
    check_executable "$SCRIPT_DIR/example_cuda_cxx_profile"
fi

if [[ "$SKIP_DRY_RUN" != "1" ]]; then
    echo
    echo "== Dry-run preview =="
    run_sweep 1 "$DRY_TIMESTAMP" "$DRY_TIMING_OUT" "$DRY_CORRECTNESS_OUT" "$DRY_ENV_OUT" "$DRY_MANIFEST_OUT"
    echo "dry-run manifest: $DRY_MANIFEST_OUT"
fi

echo
echo "== Full run =="
exec > >(tee -a "$LOG_OUT") 2>&1
print_summary
run_sweep 0 "$RUN_TIMESTAMP" "$TIMING_OUT" "$CORRECTNESS_OUT" "$ENV_OUT" "$MANIFEST_OUT"

echo
echo "full Study complete"
