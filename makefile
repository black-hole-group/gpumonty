# Source directory
SRC_DIR = src

# Build directory
BUILD_DIR = build

# Compiler and flags
CC = gcc
CFLAGS = -Wall -I/home/pedro/gsl/include -O3 -fopenmp
LDFLAGS = -lm -L/home/pedro/gsl/lib -lgsl -lgslcblas -fopenmp

# CUDA paths and flags
CUDA_PATH ?= /usr/local/cuda
CFLAGS += -I$(CUDA_PATH)/include
LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart

# NVCC compiler and flags
NVCC = nvcc
NVCCFLAGS = -arch=compute_60 -code=sm_60 --ptxas-options=-dlcm=cg --maxrregcount=255 -Xcompiler \-fopenmp -lgomp -I/home/pedro/gsl/include -c
EXTRALIBS = -lm -L /usr/local/cuda-12.4/lib64 -lstdc++ -lcudart -lcuda -L/home/pedro/gsl/lib

# Source files
SRCS = $(SRC_DIR)/grmonty.c $(SRC_DIR)/compton.c $(SRC_DIR)/init_geometry.c \
       $(SRC_DIR)/tetrads.c $(SRC_DIR)/geodesics.c $(SRC_DIR)/radiation.c \
       $(SRC_DIR)/jnu_mixed.c $(SRC_DIR)/hotcross.c \
       $(SRC_DIR)/track_super_photon.c $(SRC_DIR)/scatter_super_photon.c \
       $(SRC_DIR)/harm_model.c $(SRC_DIR)/harm_utils.c \
       $(SRC_DIR)/hamr_model.c \
       $(SRC_DIR)/init_hamr_data2D.c $(SRC_DIR)/init_harm_data.c

# GPU source file
GPU_SRC = $(SRC_DIR)/GPU_grmonty.cu

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))
GPU_OBJ = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(GPU_SRC))

# Include files
INCS = $(SRC_DIR)/decs.h $(SRC_DIR)/constants.h $(SRC_DIR)/harm_model.h $(SRC_DIR)/defs.h $(SRC_DIR)/config.h $(SRC_DIR)/gpu_header.h $(SRC_DIR)/defs_CUDA.h

# Executable
EXECUTABLE = grmonty

# Build rule
$(EXECUTABLE): $(OBJS) $(GPU_OBJ) $(INCS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $(EXECUTABLE) $(OBJS) $(GPU_OBJ) $(LDFLAGS) $(EXTRALIBS)

# Compile rule for C files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(INCS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Compile rule for GPU file
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(INCS) | $(BUILD_DIR) 
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean rule
clean:
	/bin/rm -rf $(BUILD_DIR) $(EXECUTABLE)

