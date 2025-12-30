########################################################################
# Makefile.inc (Common build configuration for PaScaL_TDMA)
#
# This file stores compiler and flag settings shared by all sub-Makefiles.
# Users can adjust compilers or optimization flags *only here* without
# touching individual Makefiles in src/ or examples/.
#
# This structure follows the original PaScaL_TDMA 2.0 design.
########################################################################

# ----------------------------------------------------------------------
# Compiler settings
# ----------------------------------------------------------------------
# FC     : MPI-capable Fortran compiler (NVHPC + OpenMPI or MPICH)
# AR     : Archiver used to create static libraries
# RANLIB : Index generator for static archives
#
# Users may modify FC to match their system environment.
# Example alternatives:
#   FC = mpif90
#   FC = mpifort
#   FC = nvfortran -Mmpi
# ----------------------------------------------------------------------
FC     = mpif90
AR     = ar
RANLIB = ranlib


# ----------------------------------------------------------------------
# Optimization flags
# ----------------------------------------------------------------------
# FLAG     : General optimization flags for CPU code paths
# CUDAFLAG : GPU-related compiler flags (assuming NVHPC/nvfortran)
#
# NOTE:
#   If your system uses additional CUDA/Fortran options (e.g., -gpu=cc80),
#   you may append them here.
#
#   These flags are automatically used by src/Makefile and examples/Makefile.
# ----------------------------------------------------------------------
FLAG      = -O3
CUDAFLAG  = -cuda


# ----------------------------------------------------------------------
# Additional customization
# ----------------------------------------------------------------------
# Users may append more application-specific or system-specific options here,
# such as include paths, debugging options, or architecture tuning flags.
#
# Example:
#   EXTRAFLAG = -gpu=cc90 -Minfo=accel
#
# These variables can be referenced later if needed.
# ----------------------------------------------------------------------
# EXTRAFLAG =
