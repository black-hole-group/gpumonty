# /*
#  * GPUmonty - makefile
#  * Copyright (C) 2026 Pedro Naethe Motta
#  *
#  * This program is free software: you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation, either version 2 of the License, or
#  * (at your option) any later version.
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program.  If not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
#  */


# Model name ( harm_model, sphere_model, iharm_model)
MODEL = iharm
MODEL_DIR = $(SRC_DIR)/$(MODEL)_model


# NEW: Toggle for automatic GPU block tuning (1 = Enable, 0 = Disable)
BLOCK_TUNING ?= 1

# CUDA auto-detection: derive CUDA root from nvcc location on PATH
CUDA_PATH ?= $(shell \
	if command -v nvcc >/dev/null 2>&1; then \
		dirname $$(dirname $$(command -v nvcc)); \
	elif [ -d /usr/local/cuda ]; then \
		echo /usr/local/cuda; \
	else \
		echo ""; \
	fi)

# GSL auto-detection: derive GSL root from gsl-config or fallback paths
GSL_HOME ?= $(strip $(shell \
	if [ -n "$$GSL_HOME" ] && [ -d "$$GSL_HOME/include/gsl" ]; then \
		echo $$GSL_HOME; \
	elif command -v gsl-config >/dev/null 2>&1; then \
		gsl-config --prefix; \
	elif [ -d "$$HOME/gsl/include/gsl" ]; then \
		echo $$HOME/gsl; \
	elif [ -d /usr/local/include/gsl ]; then \
		echo /usr/local; \
	elif [ -d /usr/include/gsl ]; then \
		echo /usr; \
	else \
		echo ""; \
	fi))

GSL_PATH ?= $(GSL_HOME)

#GSL setup
GSL_PATH ?= $(GSL_HOME)

# HDF5 auto-detection: try pkg-config, then h5cc, then common paths
HDF5_INCLUDE ?= $(shell \
	pkg-config --cflags hdf5 2>/dev/null \
	|| (command -v h5cc >/dev/null 2>&1 && h5cc -show 2>/dev/null | tr ' ' '\n' | grep '^-I' | tr '\n' ' ') \
	|| ([ -d /usr/include/hdf5/serial ] && echo "-I/usr/include/hdf5/serial") \
	|| ([ -d /usr/local/include ] && echo "-I/usr/local/include") \
	|| echo "")

HDF5_LIB ?= $(shell \
	pkg-config --libs-only-L hdf5 2>/dev/null \
	|| (command -v h5cc >/dev/null 2>&1 && h5cc -show 2>/dev/null | tr ' ' '\n' | grep '^-L' | tr '\n' ' ') \
	|| ([ -d /usr/lib/x86_64-linux-gnu/hdf5/serial ] && echo "-L/usr/lib/x86_64-linux-gnu/hdf5/serial") \
	|| ([ -d /usr/local/lib ] && echo "-L/usr/local/lib") \
	|| echo "")



# Auto detect GPU compute capability
AUTO_CC ?= 1
NVIDIA_SMI ?= nvidia-smi

ifneq ($(MAKECMDGOALS),clean)

ifeq ($(CUDA_PATH),)
$(error CUDA not found. Install the CUDA toolkit or set CUDA_PATH manually.)
endif
$(info [DETECT] CUDA_PATH    = $(CUDA_PATH))
$(info [DETECT] GSL_PATH     = $(GSL_PATH))
$(info [DETECT] HDF5_INCLUDE = $(HDF5_INCLUDE))
$(info [DETECT] HDF5_LIB     = $(HDF5_LIB))
ifeq ($(HDF5_INCLUDE),)
$(warning [DETECT] HDF5 include path not found. Build may fail. Set HDF5_INCLUDE manually if needed.)
endif

ifeq ($(AUTO_CC),1)
GPU_CC := $(shell \
	$(NVIDIA_SMI) --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1)

# Fallback: if 6.1, force 6.0
# This has been added due to weird compatibility issues with 6.1 archtecture specifically for the terminator machine
# I personally haven't tested if it happens for other machines with 6.1 archtecture, might be a driver specific issue for this machine.
ifeq ($(GPU_CC),6.1)
GPU_CC := 6.0
endif

GPU_CC_NODOT := $(subst .,,$(GPU_CC))

$(info Detected GPU compute capability: $(GPU_CC))
ARCH := compute_$(GPU_CC_NODOT)
CODE := sm_$(GPU_CC_NODOT)
CODE_LTO := lto_$(GPU_CC_NODOT)
else
# Manually set GPU compute capability here. Change GPU_CC to your GPU's compute capability.
GPU_CC := 6.0
GPU_CC_NODOT := $(subst .,,$(GPU_CC))
ARCH := compute_$(GPU_CC_NODOT)
CODE := sm_$(GPU_CC_NODOT)
CODE_LTO := lto_$(GPU_CC_NODOT)
RED_BOLD := $(shell printf "\033[1;31m")
RESET    := $(shell printf "\033[0m")
$(info $(RED_BOLD)GPU compute capability manually selected to: $(GPU_CC)$(RESET))
endif
endif

# Directories
SRC_DIR = src
BUILD_DIR = build

# CUDA setup
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_INCLUDE = -I$(CUDA_PATH)/include
CUDA_LIB_DIR := $(shell \
	if [ -d "$(CUDA_PATH)/lib64" ]; then \
		echo "$(CUDA_PATH)/lib64"; \
	else \
		echo "$(CUDA_PATH)/lib"; \
	fi)
CUDA_LIB = -L$(CUDA_LIB_DIR)

# Debug and release flags
DEBUG_FLAGS = -g -code=$(CODE)
RELEASE_FLAGS = -code=$(CODE_LTO) -dlto -O3

NVCCFLAGS_COMMON = -arch=$(ARCH) -rdc=true --use_fast_math -lineinfo --ptxas-options="-v -dlcm=cg" --maxrregcount=255\
               -Xcompiler="-fopenmp -lgomp" -I$(GSL_PATH)/include -I$(MODEL_DIR) $(HDF5_INCLUDE) -Wno-deprecated-gpu-targets

NVCCFLAGS_DEBUG =  $(NVCCFLAGS_COMMON) $(DEBUG_FLAGS)
NVCCFLAGS_RELEASE = $(NVCCFLAGS_COMMON) $(RELEASE_FLAGS)

# Choose build type (default: release)
BUILD_TYPE ?= release
ifeq ($(BUILD_TYPE),debug)
    NVCCFLAGS = $(NVCCFLAGS_DEBUG)
else
    NVCCFLAGS = $(NVCCFLAGS_RELEASE)
endif

## VERSION PRESERVATION ##
MAKEFILE_PATH := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
GIT_VERSION := $(shell cd $(MAKEFILE_PATH); git describe --dirty --always --tags)

# Linker flags
LDFLAGS =  -Xcompiler  -rdynamic $(CUDA_LIB) -lcudart -lcuda -lgomp -L$(GSL_PATH)/lib -lgsl -lgslcblas -lm -lstdc++ $(HDF5_LIB) -lhdf5_hl -lhdf5

# Source and object files
CUDA_SRC = $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(MODEL_DIR)/*.cu)

# Note: We filter out query_blocks.cu so it doesn't get compiled into the main project
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(filter-out $(SRC_DIR)/query_blocks.cu, $(wildcard $(SRC_DIR)/*.cu))) \
       $(patsubst $(MODEL_DIR)/%.cu,$(BUILD_DIR)/%.o,$(wildcard $(MODEL_DIR)/*.cu))

# Include files
INCS = $(wildcard $(SRC_DIR)/*.h) $(wildcard $(MODEL_DIR)/*.h)

# Executable
EXECUTABLE = gpumonty

### ADDED: Include the configuration logic ###
include GetGPUBlocks.mk
##############################################

# Main build rule
$(EXECUTABLE): $(OBJS) $(INCS) | $(BUILD_DIR)
	@echo "Linking libraries and building $(EXECUTABLE) executable..."
	@# We redirect stderr to stdout (2>&1) so grep can catch the info messages, for some architectures, info message spam was showing up from CURAND.
	@# grep -v hides the "no reference" spam. 
	@# The "|| true" ensures that if no spam is found, grep doesn't crash the build.
	@$(NVCC) -arch=$(ARCH) -gencode arch=$(ARCH),code=$(CODE) \
		$(if $(filter release,$(BUILD_TYPE)),-dlto) \
		-o $@ $(OBJS) $(LDFLAGS) 2>&1 | grep -v "no reference to variable" || true
	@echo "------------------------------------------------"
	@echo "Compilation done! Model: $(MODEL)."
	@echo "Build type: $(BUILD_TYPE)."
	@echo "------------------------------------------------"

### ADDED: Dependency on configuration ###
# Before compiling any .o file, run the configure_gpu_blocks target
$(OBJS): configure_gpu_blocks
##########################################


# Compile rule for CUDA files in both folders
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(INCS) | $(BUILD_DIR)
	@echo "Compiling $< ..."
	@$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDE) -DVERSION=$(GIT_VERSION) -DMODEL=$(MODEL) -c -o $@ $<

$(BUILD_DIR)/%.o: $(MODEL_DIR)/%.cu $(INCS) | $(BUILD_DIR)
	@echo "Compiling $< ..."
	@$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDE) -DVERSION=$(GIT_VERSION) -DMODEL=$(MODEL) -c -o $@ $<


# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean rule
clean:
	rm -rf $(BUILD_DIR) $(EXECUTABLE)

# Phony targets
.PHONY: clean configure_gpu_blocks
