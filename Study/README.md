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

For the CUDA C++ driver, MPI device-buffer communication is the default. Set
`MPI_MODE=host` in `run_study_sweep.sh` only when host-staging fallback is
needed.

## Study Outputs

`run_study_sweep.sh` writes one output set per run:

```text
tdma_total_profile_YYMMDD_HHMMSS.csv
tdma_correctness_YYMMDD_HHMMSS.csv
tdma_environment_YYMMDD_HHMMSS.txt
```

`tdma_total_profile_*.csv` stores every measured iteration as raw data. Do not
average or discard rows during collection. For the intended study workflow,
use `iter=0` as the first-solve correctness/warm-up iteration and analyze
`iter>=1` for stable timing.

The timing CSV schema is:

```text
solver,implementation,nranks,n1,n2,n3,nsys,nrow_min,nrow_max,iter,iterations,mpi_mode,total_s_max,total_s_avg,local_compute_s_max,pack_forward_s_max,mpi_forward_s_max,unpack_forward_s_max,reduced_compute_s_max,pack_backward_s_max,mpi_backward_s_max,unpack_backward_s_max,update_compute_s_max,compute_s_max,communication_s_max,packing_s_max
```

Both drivers measure total solve time and a phase breakdown. The total columns
include the maximum rank time and the rank-average time. The phase columns use
maximum rank time, because distributed solve time is controlled by the slowest
rank.

Aggregated phase columns:

```text
compute_s_max       = local_compute + reduced_compute + update_compute
communication_s_max = mpi_forward + mpi_backward
packing_s_max       = pack_forward + unpack_forward + pack_backward + unpack_backward
```

`tdma_correctness_*.csv` stores one first-solve solution signature per
implementation and case:

```text
solver,implementation,nranks,n1,n2,n3,nsys,nrow_min,nrow_max,mpi_mode,solution_sum,solution_l2,solution_linf,sample_z0,sample_zmid,sample_zlast,expected_value,max_abs_error_to_expected
```

The current study problem expects the solution value to be `1.0` everywhere,
so `max_abs_error_to_expected` is the quick correctness check for both the
Fortran original and the CUDA C++ port.

`tdma_environment_*.txt` records the GPU, CUDA/MPI toolchain, git revision,
and sweep settings used for the run. Keep this file with the CSV outputs when
moving results between the server and local machine.

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NP_LIST="1 2 4" \
SIZE_LIST="64,64,2048 128,128,2048 128,128,4096" \
ITERATIONS=10 \
./run_study_sweep.sh
```
