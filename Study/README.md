# PaScaL_TDMAcuda Study

This directory contains matched benchmark drivers for the original CUDA
Fortran implementation and the CUDA C++ port.

## Drivers

```text
example_fortran_profile.f90
example_cuda_cxx_profile.cu
```

The Study directory is intentionally flat. After build, the executables are
also written here:

```text
example_fortran_profile
example_cuda_cxx_profile
```

Both drivers use:

```text
[n1] [n2] [n3] [iterations] [tdma_threads] [reduced_threads]
```

Defaults:

```text
64 64 2048 10 128 128
```

## CSV

The common comparison CSV schema is:

```text
solver,implementation,nranks,n1,n2,n3,nsys,nrow_min,nrow_max,iter,iterations,mpi_mode,total_s_max,total_s_avg
```

The Fortran driver measures total solve time only. Detailed CUDA C++ phase
timing is kept in the CUDA C++ port's own profiling example.

CSV files from `run_study_sweep.sh` are written directly in this directory as:

```text
tdma_total_profile_YYMMDD_HHMMSS.csv
```
