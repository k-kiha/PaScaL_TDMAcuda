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

## Files

- `PORTING_PLAN.md`: source-level porting plan and self-review.
- `include/pascal_tdma_cuda.hpp`: public CUDA C++ API.
- `src/pascal_tdma_cuda.cu`: plan, kernels, MPI all-to-all wrapper.
- `examples/ex_tdma_zdirection.cu`: C++ version of the z-direction sample.

## Verification Status

The current workstation does not expose `nvcc` or CUDA hardware, so this port has not been compiled or run here. Build and runtime validation must be done on a CUDA-capable environment.
