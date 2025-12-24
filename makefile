# Model name (hamr_model, harm_model, sphere_model, iharm_model)
MODEL_DIR = $(SRC_DIR)/harm_model

CUDA_PATH ?= /usr/local/cuda
#GSL setup
GSL_PATH ?= /home/pedro/gsl

# HDF5 setup
HDF5_INCLUDE = -I/usr/include/hdf5/serial
HDF5_LIB = -L/usr/lib/x86_64-linux-gnu/hdf5/serial


# Compiler flags
ARCH = compute_75
CODE = sm_75
CODE_LTO = lto_75

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
			   -Xcompiler="-fopenmp -lgomp" -I$(GSL_PATH)/include -I$(MODEL_DIR) $(HDF5_INCLUDE) 

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
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.cu)) \
       $(patsubst $(MODEL_DIR)/%.cu,$(BUILD_DIR)/%.o,$(wildcard $(MODEL_DIR)/*.cu))

# Include files
INCS = $(wildcard $(SRC_DIR)/*.h) $(wildcard $(MODEL_DIR)/*.h)

# Executable
EXECUTABLE = gpumonty

# Main build rule
$(EXECUTABLE): $(OBJS) $(INCS) | $(BUILD_DIR)
	@echo "Linking libraries and building $(EXECUTABLE) executable..."
	@$(NVCC)  -arch=$(ARCH) -gencode arch=$(ARCH),code=$(CODE) $(if $(filter release,$(BUILD_TYPE)),-dlto) -o $@ $(OBJS) $(LDFLAGS)


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
.PHONY: clean
