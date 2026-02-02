# Model name (hamr_model, harm_model, sphere_model, iharm_model)
MODEL_DIR = $(SRC_DIR)/sphere_model


# NEW: Toggle for automatic GPU block tuning (1 = Enable, 0 = Disable)
BLOCK_TUNING ?= 1

CUDA_PATH ?=/usr/local/cuda/


# GSL setup
GSL_PATH ?= /home/pedro/gsl

# HDF5 setup
HDF5_INCLUDE = -I/usr/include/hdf5/serial
HDF5_LIB = -L/usr/lib/x86_64-linux-gnu/hdf5/serial



# Auto detect GPU compute capability
AUTO_CC ?= 1
NVIDIA_SMI ?= nvidia-smi

GPU_CC := $(shell \
	$(NVIDIA_SMI) --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1)

GPU_CC_NODOT := $(subst .,,$(GPU_CC))

$(info Detected GPU compute capability: $(GPU_CC))

ifeq ($(AUTO_CC),1)
    ARCH := compute_$(GPU_CC_NODOT)
    CODE := sm_$(GPU_CC_NODOT)
    CODE_LTO := lto_$(GPU_CC_NODOT)
else
    ARCH := compute_86
    CODE := sm_86
    CODE_LTO := lto_86
endif


# Directories
SRC_DIR = src
BUILD_DIR = build

# CUDA setup
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_INCLUDE = -I$(CUDA_PATH)/include
CUDA_LIB = -L$(CUDA_PATH)/lib64

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

# Linker flags
LDFLAGS = $(CUDA_LIB) -lcudart -lcuda -lgomp -L$(GSL_PATH)/lib -lgsl -lgslcblas -lm -lstdc++ $(HDF5_LIB) -lhdf5_hl -lhdf5

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
	@echo "Compilation done! Model: $(notdir $(MODEL_DIR))."
	@echo "Build type: $(BUILD_TYPE)."
	@echo "------------------------------------------------"

### ADDED: Dependency on configuration ###
# Before compiling any .o file, run the configure_gpu_blocks target
$(OBJS): configure_gpu_blocks
##########################################


# Compile rule for CUDA files in both folders
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(INCS) | $(BUILD_DIR)
	@echo "Compiling $< ..."
	@$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDE) -c -o $@ $<

$(BUILD_DIR)/%.o: $(MODEL_DIR)/%.cu $(INCS) | $(BUILD_DIR)
	@echo "Compiling $< ..."
	@$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDE) -c -o $@ $<


# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean rule
clean:
	rm -rf $(BUILD_DIR) $(EXECUTABLE)

# Phony targets
.PHONY: clean configure_gpu_blocks
