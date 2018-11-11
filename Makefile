# Requirements:
# - pgcc, version >= 18.4
# - gcc, version < 7.0 (preferably 5.4.1)
# - cuda toolkit 9.2
# - cuda drivers
#
# makefile adapted from:
# https://hiltmon.com/blog/2013/07/03/a-simple-c-plus-plus-project-structure/
#

#The GPU Compute Capability (change this according to your GPU)
GPU_CC=cc60
CC=pgcc
CCFLAGS=-fast -acc -ta=tesla:$(GPU_CC),rdc -Minfo=all -O2 -Minform=warn
CCFLAGS_FORCUDALIBS=-fast -acc -ta=tesla:$(GPU_CC),nollvm,rdc -Minfo=all -O2 -Minform=warn
LDFLAGS=-Mcuda=rdc -lgsl -lgslcblas -Mcudalib=curand
NVCC =nvcc
NVCCFLAGS = -rdc=true

GRMONTY_BASEBUILD ?= .
EXE = grmonty
SRCDIR = src
BUILDDIR = $(GRMONTY_BASEBUILD)/build
TARGET = $(GRMONTY_BASEBUILD)/bin/$(EXE)

EXCLUDE=init_harm_data.c
SRCS = $(shell find $(SRCDIR) -type f -name *.c | grep -v $(EXCLUDE))
OBJS = $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SRCS:.c=.o))
INCS = $(shell find $(SRCDIR) -type f -name *.h)

# Disable compiling for now (you can still try to compile but it probably won't
# work)
all:
	@echo "[In the process of converting OpenACC to CUDA]"
	@echo "It won't be possible to compile the code for some commits..."

$(BUILDDIR)/acc_print.o: $(SRCDIR)/acc_print.cu $(INCS) Makefile
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILDDIR)/gpu_rng.o: $(SRCDIR)/gpu_rng.c $(INCS) Makefile
	$(CC) $(CCFLAGS_FORCUDALIBS) -c $< -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.c $(INCS) Makefile
	@mkdir -p $(BUILDDIR)
	$(CC) $(CCFLAGS) -c  $< -o $@

$(TARGET): $(OBJS) $(INCS) Makefile
	$(CC) $(CCFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

.PHONY: clean
clean:
	/bin/rm -f $(TARGET); /bin/rm -rf $(GRMONTY_BASEBUILD)/build
