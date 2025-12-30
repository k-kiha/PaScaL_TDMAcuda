# PaScaL_TDMA 2.1 (CUDA Fortran / MPI)

PaScaL_TDMA is a parallel tridiagonal matrix solver designed for large-scale CFD and scientific computing applications on distributed-memory GPU clusters.

This 2.1 package provides a **minimal, self-contained CUDA Fortran + MPI implementation** with:

- A reusable **library** (`libPaScaL_TDMA.a`)
- A **3D example** solving many independent tridiagonal systems along the **z-direction**
- A **simple build system** following the PaScaL_TDMA 2.0 style (`Makefile` + `Makefile.inc`)

---

## 1. Features

- **Many-system TDMA on GPUs**  
  Solves a large number of independent tridiagonal systems in parallel.

- **Distributed-memory support (MPI)**  
  1D domain decomposition in the z-direction using MPI.

- **Multi-GPU support**  
  Each MPI rank selects a GPU using `CUDAGETDEVICECOUNT` and `CUDASETDEVICE`.  
  The implementation assumes a **CUDA-aware MPI** library.

- **Two-level algorithm**
  - Local TDMA / modified TDMA on each GPU
  - Global reduced system solved via MPI all-to-all communication

- **Simple integration**  
  Core API:
  ```fortran
  call pascal_plan_create(plan, Nsys, MPI_COMM_WORLD, myrank, nprocs, nt_tdma, nt_rdtdma)
  call pascal_solver(plan, A_d, B_d, C_d, D_d, Nsys, Nrow)
  call pascal_plan_clean(plan)
  ```

---

## 2. Directory Structure

```text
.
├── Makefile          # Top-level build driver (lib/example/all/clean)
├── Makefile.inc      # Common build configuration (compilers & flags)
├── src/
│   └── PaScaL_TDMA_cuda.f90      # Library implementation (module PaScaL_TDMA_cuda)
├── examples/
│   └── ex_tdma_zdirection.f90    # Example main program
├── lib/              # (created) Static library output directory
├── include/          # (created) Fortran module files (*.mod)
└── run/              # (created) Example executable output directory
```

* **Library source**: `src/PaScaL_TDMA_cuda.f90`
* **Example**: `examples/ex_tdma_zdirection.f90`
* **Library output**: `lib/libPaScaL_TDMA.a`
* **Example executable**: `run/a.out`

---

## 3. Requirements

* **Compiler**

  * NVIDIA HPC SDK (e.g. `nvfortran` / `mpif90`)
  * A working MPI Fortran wrapper:

    * `mpif90` or `mpifort` compatible with CUDA-aware MPI

* **CUDA**

  * CUDA-enabled GPU
  * CUDA toolkit compatible with the NVHPC version in use

* **MPI**

  * MPI implementation with CUDA-aware support (e.g. OpenMPI, MPICH with GPU support)

You may need to load appropriate modules on your system, e.g.:

```bash
module load cuda/XX.X
module load nvidia_hpc_sdk/YY.Y
module load openmpi/ZZ.Z
```

(adapt to your environment)

---

## 4. Build Instructions

All builds are driven from the **top-level directory**.

### 4.1 Configure compilers and flags

Edit `Makefile.inc` if needed:

```make
FC     = mpif90
AR     = ar
RANLIB = ranlib

FLAG      = -O3
CUDAFLAG  = -cuda
```

Examples:

* Change `FC` if your MPI wrapper has a different name.
* Add GPU architecture flags if required, e.g.:

```make
CUDAFLAG = -cuda -gpu=cc80
```

### 4.2 Build the library

```bash
make lib
```

This will:

* Compile `src/PaScaL_TDMA_cuda.f90`
* Place module files in `include/`
* Create the static library:

```text
lib/libPaScaL_TDMA.a
```

### 4.3 Build the example

```bash
make example
```

This will:

* Compile `examples/ex_tdma_zdirection.f90`
* Link against `lib/libPaScaL_TDMA.a`
* Produce the example executable:

```text
run/a.out
```

### 4.4 Build everything

```bash
make all
```

Equivalent to:

```bash
make lib
make example
```

### 4.5 Clean

```bash
# Remove object files and modules
make clean

# Remove all build artifacts (including lib/, include/, run/)
make veryclean
```

---

## 5. Running the Example

The provided example `ex_tdma_zdirection.f90`:

* Sets up a **3D grid**: `n1 × n2 × n3 = 64 × 64 × 2048`
* Performs **1D domain decomposition in z** using `para`
* Constructs a simple tridiagonal system with Dirichlet-type boundary conditions
* Calls the PaScaL-TDMA solver on the GPU(s)
* Prints a few sample values from the solution for each MPI rank

### 5.1 Example run

From the top-level directory, after `make example`:

```bash
mpirun -np 4 ./run/a.out
```

Typical output (simplified):

```text
 MPI processes:           4
 CUDA devices :           4
 Grid size    :          64          64        2048
 Subdomain z-size:        512
 Memory allocated
 Matrix system initialized
 Data copied to device
 Starting PascaL TDMA solver
 PascaL TDMA solver finished
 Rank    0--
       0       0   1.000 ...   1.000 ...   1.000
       1       1   1.000 ...   1.000 ...   1.000
       2       2   1.000 ...   1.000 ...   1.000
     ...
```

* `-np 4` must not exceed the number of available GPUs unless you intentionally oversubscribe.
* Each rank selects a GPU using:

  ```fortran
  ierr    = CUDAGETDEVICECOUNT(ngpu)
  gpurank = mod(myrank, ngpu)
  ierr    = CUDASETDEVICE(gpurank)
  ```

---

## 6. API Overview

### 6.1 Module

```fortran
use PaScaL_TDMA_cuda
```

### 6.2 Plan type

```fortran
type(ptdma_plan_cuda) :: plan
```

Contains:

* MPI communicator and rank info
* Global/local sizes for reduced and transformed systems
* Gather/scatter descriptors
* Communication buffers and offsets
* CUDA launch configurations
* Device work arrays for reduced and transformed systems

### 6.3 Plan creation

```fortran
call pascal_plan_create(plan, Nsys, MPI_COMM_WORLD, myrank, nprocs, &
                        nthread_modithomas, nthread_reduced)
```

* `Nsys`                : Number of independent systems per rank (e.g. `n1sub * n2sub`)
* `nthread_modithomas`  : Threads per block for local modified TDMA
* `nthread_reduced`     : Threads per block for reduced system TDMA

### 6.4 Solver

```fortran
call pascal_solver(plan, A_d, B_d, C_d, D_d, Nsys, Nrow)
```

* All arrays are **device arrays** (allocated with `device` attribute).
* `Nrow` is the length of each tridiagonal system (local z-extent per rank).

This call:

* Runs the local TDMA / modified TDMA kernels
* Performs MPI all-to-all communication
* Solves the global reduced system
* Updates the full solution in `D_d`

### 6.5 Clean up

```fortran
call pascal_plan_clean(plan)
```

Releases all internal buffers and descriptors associated with `plan`.

---

## 7. Extending the Example

To integrate PaScaL_TDMA into your own application:

1. **Follow the same domain decomposition pattern**
   Use `para` or your own mapping to define local z-ranges per rank.

2. **Allocate your coefficient arrays on the device**

   ```fortran
   real*8, allocatable, device :: A_d(:,:), B_d(:,:), C_d(:,:), D_d(:,:)
   ```

3. **Fill these arrays with your discretized operator**
   Build the tridiagonal coefficients and RHS for each line.

4. **Create a plan and call the solver**

   ```fortran
   call pascal_plan_create(plan, Nsys, comm, myrank, nprocs, nt1, nt2)
   call pascal_solver(plan, A_d, B_d, C_d, D_d, Nsys, Nrow)
   call pascal_plan_clean(plan)
   ```

5. **Copy back and post-process** as needed on the host.

---

## 8. License and Citation

(Adjust this section according to your actual license and publication.)

* **License**: *To be defined* (e.g. MIT, BSD, GPL, etc.)
* **Citation**: If you use PaScaL_TDMA or this CUDA Fortran implementation in a publication,
  please cite the corresponding paper or repository as specified by the project maintainers.

---

## 9. Contact / Issues

* For bug reports, questions, or contributions, please use the issue tracker of the GitHub repository where this code is hosted.
* When reporting issues, please include:

  * Compiler and MPI versions
  * GPU model and driver version
  * Exact compilation commands (`make` outputs)
  * The `mpirun` command line you used

---
