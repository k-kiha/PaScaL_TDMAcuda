# PaScaL_TDMAcuda Study Repository

This repository is organized to study a CUDA Fortran + MPI tridiagonal solver
and its CUDA C++ port side by side.

## Layout

```text
Fortran_Original/  Original CUDA Fortran + MPI implementation
CUDA_CXX_Port/     CUDA C++ / MPI port
Study/             Matched drivers, sweep scripts, and CSV results
scripts/           Repository maintenance helpers
```

The purpose of this layout is to keep the original solver, the CUDA C++ port,
and the comparison experiments separate.

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
NP_LIST="1 2 4" \
SIZE_LIST="64,64,2048 128,128,2048 128,128,4096" \
ITERATIONS=10 \
./Study/run_study_sweep.sh
```

Study timing, correctness, and environment files are written directly under
`Study/`.

Set `MPI_MODE=host` explicitly if the CUDA-aware MPI device path is not usable
on a target system.

## Cleanup For Sync

Before pushing or pulling between the GPU server and local machine:

```bash
scripts/clean_for_sync.sh
scripts/clean_for_sync.sh --apply
```

The cleanup script preserves CSV files.
