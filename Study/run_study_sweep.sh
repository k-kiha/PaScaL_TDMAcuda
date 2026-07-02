#!/usr/bin/env bash
set -euo pipefail

MPIRUN=${MPIRUN:-mpirun}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORTRAN_EXE=${FORTRAN_EXE:-"$SCRIPT_DIR/example_fortran_profile"}
CXX_EXE=${CXX_EXE:-"$SCRIPT_DIR/example_cuda_cxx_profile"}
NP_LIST=${NP_LIST:-"1 2 4"}
SIZE_LIST=${SIZE_LIST:-"64,64,2048 128,128,2048 128,128,4096"}
ITERATIONS=${ITERATIONS:-10}
TDMA_THREADS=${TDMA_THREADS:-128}
REDUCED_THREADS=${REDUCED_THREADS:-128}
MPI_MODE=${MPI_MODE:-host}
RUN_FORTRAN=${RUN_FORTRAN:-1}
RUN_CXX=${RUN_CXX:-1}
OUT=${OUT:-"$SCRIPT_DIR/tdma_total_profile_$(date +%y%m%d_%H%M%S).csv"}

mkdir -p "$(dirname "$OUT")"

append_csv() {
    if [[ ! -s "$OUT" ]]; then
        tee -a "$OUT"
    else
        awk 'NR == 1 && /^solver,/ { next } { print }' | tee -a "$OUT"
    fi
}

for np in $NP_LIST; do
    for size in $SIZE_LIST; do
        IFS=',' read -r n1 n2 n3 <<< "$size"
        echo "running np=$np n1=$n1 n2=$n2 n3=$n3 iterations=$ITERATIONS" >&2

        if [[ "$RUN_FORTRAN" == "1" ]]; then
            "$MPIRUN" -np "$np" "$FORTRAN_EXE" "$n1" "$n2" "$n3" "$ITERATIONS" "$TDMA_THREADS" "$REDUCED_THREADS" \
                | append_csv
        fi

        if [[ "$RUN_CXX" == "1" ]]; then
            if [[ "$MPI_MODE" == "default" ]]; then
                "$MPIRUN" -np "$np" "$CXX_EXE" "$n1" "$n2" "$n3" "$ITERATIONS" "$TDMA_THREADS" "$REDUCED_THREADS" \
                    | append_csv
            else
                PASCAL_TDMA_MPI_MODE="$MPI_MODE" \
                "$MPIRUN" -np "$np" "$CXX_EXE" "$n1" "$n2" "$n3" "$ITERATIONS" "$TDMA_THREADS" "$REDUCED_THREADS" \
                    | append_csv
            fi
        fi
    done
done

echo "wrote $OUT" >&2
