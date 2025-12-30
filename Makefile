########################################################################
# PaScaL_TDMA 2.1 - Top-Level Makefile
#
# This Makefile does not perform compilation directly. Instead, it 
# delegates the build process to the Makefiles inside:
#     - src/       : builds the PaScaL_TDMA CUDA library
#     - examples/  : builds example executables
#
# Available targets:
#   make lib       -> Build the static library  (lib/libPaScaL_TDMA.a)
#   make example   -> Build the example binary (run/a.out)
#   make all       -> Build both library and example
#   make clean     -> Clean intermediate files in src/ and examples/
#   make veryclean -> Clean everything including lib/, include/, run/
#
# This design follows the modular structure of PaScaL_TDMA 2.0,
# allowing each subdirectory to control its own build rules.
########################################################################

include Makefile.inc

.PHONY: all lib example clean veryclean

# ----------------------------------------------------------------------
# Build both the library and example programs
# ----------------------------------------------------------------------
all: lib example


# ----------------------------------------------------------------------
# Build the static library by invoking src/Makefile
# ----------------------------------------------------------------------
lib:
	$(MAKE) -C src lib


# ----------------------------------------------------------------------
# Build example executables by invoking examples/Makefile
# ----------------------------------------------------------------------
example:
	$(MAKE) -C examples example


# ----------------------------------------------------------------------
# Remove object and module files generated during compilation
# ----------------------------------------------------------------------
clean:
	$(MAKE) -C src clean
	$(MAKE) -C examples clean
	rm -f ./lib/* ./include/* ./run/*


# ----------------------------------------------------------------------
# Remove all build artifacts (full cleanup)
# ----------------------------------------------------------------------
veryclean: clean
