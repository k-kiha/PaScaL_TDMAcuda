########################################################################
# src/Makefile : Build the PaScaL_TDMA CUDA library
#
# This Makefile compiles the CUDA-enabled TDMA solver components and
# produces a static library:
#
#     lib/libPaScaL_TDMA.a
#
# The resulting modules (*.mod) are placed in ../include/
# to be shared with example programs and external applications.
#
# This structure follows the modular design introduced in PaScaL_TDMA 2.0.
########################################################################

include ../Makefile.inc

# ----------------------------------------------------------------------
# Directory configuration
#   LIBDIR : Output directory for the static library
#   INCDIR : Directory for Fortran module files (*.mod)
# ----------------------------------------------------------------------
LIBDIR  = ../lib
INCDIR  = ../include

# ----------------------------------------------------------------------
# Library name and source objects
# ----------------------------------------------------------------------
LIBNAME = $(LIBDIR)/libPaScaL_TDMA.a
OBJS    = PaScaL_TDMA_cuda.o

.PHONY: all lib clean

# ----------------------------------------------------------------------
# Default target: build the library
# ----------------------------------------------------------------------
all: lib


# ----------------------------------------------------------------------
# Build the static library
# ----------------------------------------------------------------------
lib: $(LIBNAME)

$(LIBNAME): $(OBJS)
	mkdir -p $(LIBDIR) $(INCDIR)
	$(AR) rcs $@ $^
	$(RANLIB) $@


# ----------------------------------------------------------------------
# Object file compilation rule
#   - Places .mod files into ../include/
#   - Uses the compiler settings defined in Makefile.inc
#
# Example:
#   PaScaL_TDMA_cuda.f90 → PaScaL_TDMA_cuda.o + ../include/*.mod
# ----------------------------------------------------------------------
%.o: %.f90
	mkdir -p $(INCDIR)
	$(FC) $(FLAG) $(CUDAFLAG) -c $< -o $@ -module $(INCDIR)


# ----------------------------------------------------------------------
# Cleanup rule
# ----------------------------------------------------------------------
clean:
	rm -f *.o *.mod
