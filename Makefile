#
#
# IMPORTANT:
# - You must have the env variable CUDA_HOME set to your CUDA
# instalation
# - If you have a CUDA version different than cuda9.2, you must set the
# env variable CUDA_VERSION with the respectively CUDA version of your CUDA_HOME
# - If you have a NVIDIA GPU with compute capability different than 6.0, you
# must set the env var COMPUTE_CAP to it. (Format: 6.0 -> 60)
#
# Requirements:
# - pgcc, version >= 18.4
# - cuda toolkit
#
# makefile adapted from:
# https://hiltmon.com/blog/2013/07/03/a-simple-c-plus-plus-project-structure/
#

COMPUTE_CAP ?= 60
CUDA_VERSION ?= cuda9.2

CC=pgcc
CCFLAGS=-fast -acc -ta=tesla:cc$(COMPUTE_CAP),$(CUDA_VERSION),rdc -Minfo=, -O3 -Minform=warn
LDFLAGS=-Mcuda=rdc -lgsl -lgslcblas -Mcudalib=curand

NVCC=nvcc
NVCCFLAGS=-rdc=true -O3 -arch=sm_$(COMPUTE_CAP)

GRMONTY_BASEBUILD ?= .
EXE = grmonty
SRCDIR = src
BUILDDIR = $(GRMONTY_BASEBUILD)/build
TARGET = $(GRMONTY_BASEBUILD)/bin/$(EXE)

SRCS_C = $(shell find $(SRCDIR) -type f -name *.c)
SRCS_CU = $(shell find $(SRCDIR) -type f -name *.cu)
OBJS_C = $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SRCS_C:.c=.o))
OBJS_CU = $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SRCS_CU:.cu=.o))
INCS = $(shell find $(SRCDIR) -type f -name *.h)

all: $(TARGET)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu $(INCS) Makefile
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.c $(INCS) Makefile
	@mkdir -p $(BUILDDIR)
	$(CC) $(CCFLAGS) -c  $< -o $@

$(TARGET): $(OBJS_C) $(OBJS_CU) $(INCS) Makefile
	$(CC) $(CCFLAGS) $(OBJS_C) $(OBJS_CU) -o $(TARGET) $(LDFLAGS)


.PHONY: clean
clean:
	/bin/rm -f $(TARGET); /bin/rm -rf $(GRMONTY_BASEBUILD)/build
