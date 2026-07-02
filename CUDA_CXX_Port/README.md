# PaScaL_TDMAcuda CUDA C++ Port

This directory contains a CUDA C++ port of
`../Fortran_Original/src/PaScaL_TDMA_cuda.f90`. The port keeps the original
MPI-parallel TDMA solver flow and exposes a C++/CUDA interface suitable for
matched validation and performance studies.

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

Requires the NVIDIA CUDA Toolkit, `nvcc`, and MPI.

```bash
make CUDA_ARCH=90
```

`CUDA_ARCH=90` matches the H200 validation system used for the current study.
Set this value for the target GPU architecture when building elsewhere. If your
MPI wrapper is not `mpicxx`:

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
mpirun -np 4 ./run/ex_tdma_profile 64 64 2048 10
```

Arguments:

```text
ex_tdma_profile [n1] [n2] [n3] [iterations] [tdma_threads] [reduced_threads]
```

For CUDA C++ multi-case studies, use the Linux sweep script:

```bash
NP_LIST="1 2 4 8" \
SIZE_LIST="64,64,2048 128,128,4096" \
ITERATIONS=10 \
./scripts/run_tdma_profile_sweep.sh
```

The script writes CSV files under `profile_results/`.
It uses `MPI_MODE=device` by default. Set `MPI_MODE=host` explicitly to use
host-staging fallback.

For matched Fortran/CUDA C++ comparison studies, use `../Study`.

## Files

- `PORTING_PLAN.md`: source-level porting plan and self-review.
- `include/pascal_tdma_cuda.hpp`: public CUDA C++ API.
- `src/pascal_tdma_cuda.cu`: plan, kernels, MPI all-to-all wrapper.
- `examples/ex_tdma_zdirection.cu`: C++ version of the z-direction sample.
- `examples/ex_tdma_profile.cu`: one-case timing example with CSV output.
- `scripts/run_tdma_profile_sweep.sh`: Linux helper for rank/size sweeps.

## Verification Status

The port has been built and exercised on an H200 multi-GPU server with CUDA
12.9 and Open MPI. The matched study report in `../Study` summarizes the
current correctness and performance evidence. Local machines without CUDA
hardware can still inspect and edit the source, but build and runtime
validation require a CUDA-capable environment.
