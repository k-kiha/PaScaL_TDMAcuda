NVCC      ?= nvcc
MPICXX    ?= mpicxx
CUDA_ARCH ?= 90
OPT       ?= -O3

SRC_DIR     := src
INC_DIR     := include
EXAMPLE_DIR := examples
BUILD_DIR   := build
LIB_DIR     := lib
RUN_DIR     := run

GENCODE := -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
NVCCFLAGS := -std=c++17 $(OPT) $(GENCODE) -ccbin $(MPICXX) -I$(INC_DIR)

LIB := $(LIB_DIR)/libpascal_tdma_cuda.a
OBJ := $(BUILD_DIR)/pascal_tdma_cuda.o
EXE := $(RUN_DIR)/ex_tdma_zdirection
PROFILE_EXE := $(RUN_DIR)/ex_tdma_profile

.PHONY: all lib example profile clean veryclean dirs

all: lib example

lib: $(LIB)

example: $(EXE) $(PROFILE_EXE)

profile: $(PROFILE_EXE)

dirs:
	mkdir -p $(BUILD_DIR) $(LIB_DIR) $(RUN_DIR)

$(OBJ): $(SRC_DIR)/pascal_tdma_cuda.cu $(INC_DIR)/pascal_tdma_cuda.hpp | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(LIB): $(OBJ) | dirs
	ar rcs $@ $<
	ranlib $@

$(EXE): $(EXAMPLE_DIR)/ex_tdma_zdirection.cu $(LIB) | dirs
	$(NVCC) $(NVCCFLAGS) $< $(LIB) -o $@

$(PROFILE_EXE): $(EXAMPLE_DIR)/ex_tdma_profile.cu $(LIB) | dirs
	$(NVCC) $(NVCCFLAGS) $< $(LIB) -o $@

clean:
	rm -rf $(BUILD_DIR)

veryclean: clean
	rm -rf $(LIB_DIR) $(RUN_DIR)
