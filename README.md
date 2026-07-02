# CUDA C++ Port and Multi-GPU Study of PaScaL TDMA

This repository keeps the original CUDA Fortran + MPI TDMA solver, a CUDA C++
port, and matched benchmark drivers side by side. The goal is to preserve the
solver's MPI-parallel algorithmic structure while making the port, validation,
and multi-GPU performance analysis easy to inspect.

## Publication Background

This repository originated as the developer repository associated with the
PaScaL_TDMA 2.1 paper in Computer Physics Communications. The CUDA Fortran
implementation in `Fortran_Original/` is the published reference code, while
`CUDA_CXX_Port/` and `Study/` extend the repository with a CUDA C++ port and a
matched validation/performance workflow.

Reference:

K.-H. Kim, D. Lee, J. Lee, S. Oh, S. Lee, J.-H. Kang, and J.-I. Choi,
"PaScaL_TDMA 2.1: A register-resident multi-GPU tridiagonal matrix solver with
optimized communication for large-scale CFD simulations,"
Computer Physics Communications 323 (2026) 110120.
https://doi.org/10.1016/j.cpc.2026.110120

## Layout

```text
Fortran_Original/  Original CUDA Fortran + MPI implementation
CUDA_CXX_Port/     CUDA C++ / MPI port of the original solver flow
Study/             Matched drivers, sweep script, report, and result data
scripts/           Repository maintenance helpers
```

The layout separates the reference implementation, the CUDA C++ port, and the
comparison study so that code changes and benchmark evidence can be reviewed
independently.

## Build

On the H200 server, use `CUDA_ARCH=90`.

```bash
make all CUDA_ARCH=90
```

Build only the original CUDA Fortran implementation:

```bash
make fortran FC=mpifort
```

Build only the CUDA C++ port:

```bash
make cuda-cxx CUDA_ARCH=90
```

Build only matched Study drivers after both libraries exist:

```bash
make study CUDA_ARCH=90
```

## Run Existing Samples

Original CUDA Fortran sample:

```bash
cd Fortran_Original
mpirun -np 4 ./run/a.out
```

CUDA C++ sample:

```bash
cd CUDA_CXX_Port
mpirun -np 4 ./run/ex_tdma_zdirection
```

The CUDA C++ port uses CUDA-aware MPI device-buffer communication by default.
Use `PASCAL_TDMA_MPI_MODE=host` only when host-staging fallback is needed.

## Run Matched Study Drivers

From the repository root:

```bash
mpirun -np 4 ./Study/example_cuda_cxx_profile 64 64 2048 10
mpirun -np 4 ./Study/example_fortran_profile 64 64 2048 10
```

Both Study drivers report total solver time and phase breakdown columns. The
CUDA C++ port also has a lower-level profiling example under
`CUDA_CXX_Port/examples/ex_tdma_profile.cu`.

For multiple cases:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
STUDY_PRESET=custom \
NP_LIST="1 2 4" \
SIZE_LIST="64,64,2048 128,128,2048 128,128,4096" \
ITERATIONS=10 \
./Study/run_study_sweep.sh
```

Omit `STUDY_PRESET=custom` to run the prepared benchmark study matrix.

New sweep runs write timing, correctness, manifest, and environment files under
`Study/` by default. Curated report inputs and generated figures/tables are kept
under `Study/result/`.

Set `MPI_MODE=host` explicitly if the CUDA-aware MPI device path is not usable
on a target system.

## Cleanup For Sync

Before pushing or pulling between the GPU server and local machine:

```bash
scripts/clean_for_sync.sh
scripts/clean_for_sync.sh --apply
```

The cleanup script removes generated binaries and intermediate build outputs.
It preserves CSV data and the curated report assets under `Study/result/`.
