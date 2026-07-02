#!/usr/bin/env bash
set -euo pipefail

MPIRUN=${MPIRUN:-mpirun}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FORTRAN_EXE=${FORTRAN_EXE:-"$SCRIPT_DIR/example_fortran_profile"}
CXX_EXE=${CXX_EXE:-"$SCRIPT_DIR/example_cuda_cxx_profile"}
NP_LIST=${NP_LIST:-"1 2 4"}
SIZE_LIST=${SIZE_LIST:-"64,64,2048 128,128,2048 128,128,4096"}
ITERATIONS=${ITERATIONS:-10}
TDMA_THREADS=${TDMA_THREADS:-128}
REDUCED_THREADS=${REDUCED_THREADS:-128}
MPI_MODE=${MPI_MODE:-device}
RUN_FORTRAN=${RUN_FORTRAN:-1}
RUN_CXX=${RUN_CXX:-1}
TIMESTAMP=${TIMESTAMP:-$(date +%y%m%d_%H%M%S)}
OUT=${OUT:-"$SCRIPT_DIR/tdma_total_profile_${TIMESTAMP}.csv"}
CORRECTNESS_OUT=${CORRECTNESS_OUT:-"$SCRIPT_DIR/tdma_correctness_${TIMESTAMP}.csv"}
ENV_OUT=${ENV_OUT:-"$SCRIPT_DIR/tdma_environment_${TIMESTAMP}.txt"}

mkdir -p "$(dirname "$OUT")"
mkdir -p "$(dirname "$CORRECTNESS_OUT")"
mkdir -p "$(dirname "$ENV_OUT")"

append_csv() {
    if [[ ! -s "$OUT" ]]; then
        tee -a "$OUT"
    else
        awk 'NR == 1 && /^solver,/ { next } { print }' | tee -a "$OUT"
    fi
}

capture_environment() {
    {
        echo "# PaScaL_TDMAcuda Study Environment"
        echo "date=$(date '+%Y-%m-%dT%H:%M:%S%z')"
        echo "hostname=$(hostname)"
        echo "pwd=$PWD"
        echo "root_dir=$ROOT_DIR"
        echo "script_dir=$SCRIPT_DIR"
        echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
        echo "np_list=$NP_LIST"
        echo "size_list=$SIZE_LIST"
        echo "iterations=$ITERATIONS"
        echo "tdma_threads=$TDMA_THREADS"
        echo "reduced_threads=$REDUCED_THREADS"
        echo "mpi_mode=$MPI_MODE"
        echo "run_fortran=$RUN_FORTRAN"
        echo "run_cxx=$RUN_CXX"
        echo "timing_csv=$OUT"
        echo "correctness_csv=$CORRECTNESS_OUT"
        echo "environment_file=$ENV_OUT"
        echo
        echo "## git"
        git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || true
        git -C "$ROOT_DIR" status --short 2>/dev/null || true
        echo
        echo "## nvidia-smi"
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi || true
            echo
            echo "## nvidia-smi topo -m"
            nvidia-smi topo -m || true
        else
            echo "nvidia-smi not found"
        fi
        echo
        echo "## nvcc --version"
        if command -v nvcc >/dev/null 2>&1; then
            nvcc --version || true
        else
            echo "nvcc not found"
        fi
        echo
        echo "## mpirun --version"
        "$MPIRUN" --version || true
        echo
        echo "## mpifort --version"
        if command -v mpifort >/dev/null 2>&1; then
            mpifort --version || true
        else
            echo "mpifort not found"
        fi
        echo
        echo "## mpicxx --version"
        if command -v mpicxx >/dev/null 2>&1; then
            mpicxx --version || true
        else
            echo "mpicxx not found"
        fi
    } > "$ENV_OUT"
}

capture_environment

for np in $NP_LIST; do
    for size in $SIZE_LIST; do
        IFS=',' read -r n1 n2 n3 <<< "$size"
        echo "running np=$np n1=$n1 n2=$n2 n3=$n3 iterations=$ITERATIONS" >&2

        if [[ "$RUN_FORTRAN" == "1" ]]; then
            PASCAL_TDMA_CORRECTNESS_OUT="$CORRECTNESS_OUT" \
            "$MPIRUN" -np "$np" "$FORTRAN_EXE" "$n1" "$n2" "$n3" "$ITERATIONS" "$TDMA_THREADS" "$REDUCED_THREADS" \
                | append_csv
        fi

        if [[ "$RUN_CXX" == "1" ]]; then
            if [[ "$MPI_MODE" == "default" ]]; then
                PASCAL_TDMA_CORRECTNESS_OUT="$CORRECTNESS_OUT" \
                "$MPIRUN" -np "$np" "$CXX_EXE" "$n1" "$n2" "$n3" "$ITERATIONS" "$TDMA_THREADS" "$REDUCED_THREADS" \
                    | append_csv
            else
                PASCAL_TDMA_CORRECTNESS_OUT="$CORRECTNESS_OUT" \
                PASCAL_TDMA_MPI_MODE="$MPI_MODE" \
                "$MPIRUN" -np "$np" "$CXX_EXE" "$n1" "$n2" "$n3" "$ITERATIONS" "$TDMA_THREADS" "$REDUCED_THREADS" \
                    | append_csv
            fi
        fi
    done
done

echo "wrote $OUT" >&2
echo "wrote $CORRECTNESS_OUT" >&2
echo "wrote $ENV_OUT" >&2
