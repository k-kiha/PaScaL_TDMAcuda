# PaScaL_TDMAcuda

CUDA Fortran and CUDA C++ implementations of the PaScaL_TDMA 2.1
multi-GPU non-cyclic tridiagonal matrix solver, together with corresponding profiling drivers
and a reproducible study workflow.

PaScaL_TDMA solves many independent tridiagonal systems whose row direction is
distributed across MPI ranks. For more than one rank, each GPU first reduces
its local row segment, the reduced interface systems are redistributed with
`MPI_Alltoallv`, and the full local solutions are reconstructed after the
global reduced solve.

## Publication and code archive

The solver is described in:

> Ki-Ha Kim, Dongjin Lee, Junhwan Lee, Sehyeong Oh, Seungwon Lee, Ji-Hoon
> Kang, and Jung-Il Choi, “PaScaL_TDMA 2.1: A register-resident multi-GPU
> tridiagonal matrix solver with optimized communication for large-scale CFD
> simulations,” *Computer Physics Communications* 323 (2026) 110120.
> <https://doi.org/10.1016/j.cpc.2026.110120>

The publication snapshot is archived as Mendeley Data version 3:
<https://data.mendeley.com/datasets/49z6fh94z3/3>.

This Git repository is the maintained development version. It is not expected
to be byte-identical to the archived snapshot: the CUDA Fortran source includes
a post-archive nonblocking-request fix (`d69ae95`) and later profiling support
(`902705e`), while the CUDA C++ port and `Study/` workflow are repository
extensions.

## Repository layout

```text
Fortran_Original/  CUDA Fortran + MPI implementation
CUDA_CXX_Port/     CUDA C++17 + MPI port
Study/             Corresponding profile drivers, sweep workflow, report, and data
scripts/           Repository maintenance helpers
```

Component-specific instructions are available in
[`Fortran_Original/README.md`](Fortran_Original/README.md),
[`CUDA_CXX_Port/README.md`](CUDA_CXX_Port/README.md), and
[`Study/README.md`](Study/README.md).

## Requirements

- Linux or a compatible GPU cluster environment
- NVIDIA GPU and driver
- CUDA Toolkit with `nvcc` for the CUDA C++ implementation
- NVIDIA HPC SDK CUDA Fortran compiler for the Fortran implementation
- MPI C++ and Fortran compiler wrappers
- GNU Make
- CUDA-aware MPI for the CUDA Fortran implementation and the default CUDA C++
  communication path

The checked-in defaults target the H200 validation environment
(`CUDA_ARCH=90`). Override the compiler wrappers and architecture for another
system.

## Build from the repository root

```bash
make all CUDA_ARCH=90 FC=mpifort MPICXX=mpicxx
```

The root targets have the following exact scope:

| Target | Output |
| --- | --- |
| `make all` | both libraries and both `Study/` profile drivers |
| `make libs` | both libraries only |
| `make fortran` | CUDA Fortran library and `Fortran_Original/run/a.out` |
| `make cuda-cxx` | CUDA C++ library and both C++ examples |
| `make cuda-cxx-profile` | CUDA C++ library and profiling example |
| `make study` | both libraries and both corresponding `Study/` profile drivers |

For example, a complete standalone-example build is:

```bash
make fortran FC=mpifort CUDA_ARCH=90
make cuda-cxx MPICXX=mpicxx CUDA_ARCH=90
```

## Run the examples

CUDA Fortran:

```bash
mpirun -np 4 ./Fortran_Original/run/a.out
```

CUDA C++:

```bash
mpirun -np 4 ./CUDA_CXX_Port/run/ex_tdma_zdirection
```

The examples select a GPU using `rank % visible_device_count`. Normally, the
number of ranks placed on a node should not exceed the number of GPUs made
visible to those ranks.

## MPI buffer modes

| Implementation | Default | Host-staging fallback |
| --- | --- | --- |
| CUDA Fortran | device-buffer CUDA-aware MPI | not provided |
| CUDA C++ | device-buffer CUDA-aware MPI | `PASCAL_TDMA_MPI_MODE=host` |

For a non-CUDA-aware MPI implementation, only the CUDA C++ path can explicitly
stage communication through host memory:

```bash
PASCAL_TDMA_MPI_MODE=host \
  mpirun -np 4 ./CUDA_CXX_Port/run/ex_tdma_zdirection
```

Only the exact value `host` selects host staging. An unset variable or any
other value selects the device-direct path.

## Solver data contract

Both implementations operate on double-precision device arrays representing
`nsys` independent systems of local length `nrow`. The contiguous dimension is
the system index:

```text
offset(sys, row) = sys + row * nsys
```

`D` contains the right-hand side on entry and the solution on return. The
solver also uses coefficient arrays as in-place workspace:

| MPI ranks | Arrays modified by a solve |
| --- | --- |
| 1 | `C`, `D` |
| more than 1 | `A`, `C`, `D` |

`B` is read but not modified. Restore the original coefficients and right-hand
side before solving the same original system again. Reusing only the plan is
supported; reusing already transformed input arrays is not equivalent to a new
solve of the original system.

Additional operational constraints are:

- `nsys > 0`;
- for a multi-rank solve, every rank must have `nrow >= 2`;
- the current reduced-system partition requires `nsys >= number of ranks`;
- a global row length is partitioned with floor/ceiling balance, so local
  `nrow` values can differ by one when it is not divisible by the rank count.

## Study workflow

Build the corresponding drivers:

```bash
make study CUDA_ARCH=90 FC=mpifort MPICXX=mpicxx
```

Run one case directly:

```bash
mpirun -np 4 ./Study/example_cuda_cxx_profile 64 64 2048 10
mpirun -np 4 ./Study/example_fortran_profile 64 64 2048 10
```

Run the prepared sweep from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
STUDY_PRESET=quick \
./Study/run_study_sweep.sh
```

For CUDA C++ only on a non-CUDA-aware MPI stack, force host mode in both the
default and comparison case lists:

```bash
RUN_FORTRAN=0 \
MPI_MODE=host \
MPI_MODE_LIST=host \
STUDY_PRESET=quick \
./Study/run_study_sweep.sh
```

The current curated report dataset contains 25 CUDA C++ cases and no Fortran
timing rows. Correctness is recorded only after iteration 0. In the current
profiling drivers, input arrays are initialized once and then passed to the
in-place solver repeatedly; consequently, iterations 1–9 measure repeated
execution on already transformed arrays, not independent solves of the same
original system. See [`Study/README.md`](Study/README.md) and the
[`Study report`](Study/report_cuda_cxx_porting_tdma_solver_study.md) before
interpreting the timing tables.

## Cleanup

Standard Make cleanup:

```bash
make clean
make veryclean
```

To preview repository-wide generated-file cleanup while preserving CSV data
and curated report assets:

```bash
scripts/clean_for_sync.sh
scripts/clean_for_sync.sh --apply
```

The second command removes the paths printed by the dry run, so inspect the
preview first.

## Citation

GitHub and citation tools can read [`CITATION.cff`](CITATION.cff). A BibTeX
entry for the PaScaL_TDMA 2.1 paper is:

```bibtex
@article{Kim2026PaScaLTDMA21,
  author  = {Kim, Ki-Ha and Lee, Dongjin and Lee, Junhwan and Oh, Sehyeong and
             Lee, Seungwon and Kang, Ji-Hoon and Choi, Jung-Il},
  title   = {PaScaL_TDMA 2.1: A register-resident multi-GPU tridiagonal matrix
             solver with optimized communication for large-scale CFD simulations},
  journal = {Computer Physics Communications},
  volume  = {323},
  pages   = {110120},
  year    = {2026},
  doi     = {10.1016/j.cpc.2026.110120}
}
```

## License

This repository is distributed under the [MIT License](LICENSE).
