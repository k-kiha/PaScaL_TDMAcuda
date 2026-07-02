# PaScaL_TDMAcuda CUDA C++ Port

This directory is a first-pass CUDA C++ port of `../src/PaScaL_TDMA_cuda.f90`.

## Scope

- Preserves the original PaScaL TDMA flow.
- Uses CUDA C++ runtime API and MPI C++ wrapper.
- Keeps Fortran-compatible flat layout:

```cpp
A[sys + row * nsys]
```

- Uses CUDA-aware MPI by default.
- Provides a host-staging fallback for non CUDA-aware MPI:

```bash
export PASCAL_TDMA_MPI_MODE=host
```

## Build

Requires NVIDIA CUDA Toolkit, `nvcc`, and MPI.

```bash
make CUDA_ARCH=80
```

If your MPI wrapper is not `mpicxx`:

```bash
make MPICXX=/path/to/mpicxx CUDA_ARCH=90
```

## Run

```bash
mpirun -np 4 ./run/ex_tdma_zdirection
```

For non CUDA-aware MPI:

```bash
PASCAL_TDMA_MPI_MODE=host mpirun -np 4 ./run/ex_tdma_zdirection
```

## Profiling

`ex_tdma_profile` reuses one plan and repeats only the solver call for one
problem size. The first iteration is kept in the CSV output so it can be treated
as warm-up during analysis.

```bash
make profile CUDA_ARCH=90
PASCAL_TDMA_MPI_MODE=host mpirun -np 4 ./run/ex_tdma_profile 64 64 2048 10
```

Arguments:

```text
ex_tdma_profile [n1] [n2] [n3] [iterations] [tdma_threads] [reduced_threads]
```

For multi-case studies, use the Linux sweep script:

```bash
NP_LIST="1 2 4 8" \
SIZE_LIST="64,64,2048 128,128,4096" \
ITERATIONS=10 \
MPI_MODE=host \
./scripts/run_tdma_profile_sweep.sh
```

The script writes CSV files under `profile_results/`.

## Files

- `PORTING_PLAN.md`: source-level porting plan and self-review.
- `include/pascal_tdma_cuda.hpp`: public CUDA C++ API.
- `src/pascal_tdma_cuda.cu`: plan, kernels, MPI all-to-all wrapper.
- `examples/ex_tdma_zdirection.cu`: C++ version of the z-direction sample.
- `examples/ex_tdma_profile.cu`: one-case timing example with CSV output.
- `scripts/run_tdma_profile_sweep.sh`: Linux helper for rank/size sweeps.

## Verification Status

The current workstation does not expose `nvcc` or CUDA hardware, so this port has not been compiled or run here. Build and runtime validation must be done on a CUDA-capable environment.
