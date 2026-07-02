include Makefile.inc

.PHONY: all lib example clean veryclean dirs help

all: lib example

dirs:
	mkdir -p lib include run

lib: dirs
	$(MAKE) -C src lib

example: lib
	$(MAKE) -C examples example

clean:
	$(MAKE) -C src clean
	$(MAKE) -C examples clean
	rm -f lib/* include/*
	find run -type f ! -name 'job.sh' -delete

veryclean: clean

help:
	@echo "Original CUDA Fortran PaScaL_TDMA build"
	@echo "  make lib       build lib/libPaScaL_TDMA.a and include/*.mod"
	@echo "  make example   build run/a.out"
	@echo "  make all       build library and original sample"
