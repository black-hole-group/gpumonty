# Model name
MODEL_DIR = $(SRC_DIR)/sphere_model

# Directories
SRC_DIR = src
BUILD_DIR = build

# CUDA setup
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_INCLUDE = -I$(CUDA_PATH)/include
CUDA_LIB = -L$(CUDA_PATH)/lib64

# Compiler flags
NVCCFLAGS = -G -arch=compute_60 -code=sm_60 -rdc=true --ptxas-options=-dlcm=cg --maxrregcount=255 -Xcompiler="-fopenmp"
NVCCFLAGS += -I/home/pedro/gsl/include -O3
NVCCFLAGS += -I$(MODEL_DIR)  # Add model directory to the include path

# Linker flags
LDFLAGS = $(CUDA_LIB) -lcudart -lcuda -lgomp
LDFLAGS += -L/home/pedro/gsl/lib -lgsl -lgslcblas
LDFLAGS += -lm -lstdc++

# Source files
CUDA_SRC = $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(MODEL_DIR)/*.cu)

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.cu)) \
       $(patsubst $(MODEL_DIR)/%.cu,$(BUILD_DIR)/%.o,$(wildcard $(MODEL_DIR)/*.cu))

# Include files
INCS = $(wildcard $(SRC_DIR)/*.h) $(wildcard $(MODEL_DIR)/*.h)

# Executable
EXECUTABLE = gpumonty

# Main build rule
$(EXECUTABLE): $(OBJS) $(INCS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS) $(LDFLAGS)

# Compile rule for CUDA files in both folders
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(INCS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDE) -c -o $@ $<

$(BUILD_DIR)/%.o: $(MODEL_DIR)/%.cu $(INCS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDE) -c -o $@ $<

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean rule
clean:
	rm -rf $(BUILD_DIR) $(EXECUTABLE)

# Phony targets
.PHONY: clean
