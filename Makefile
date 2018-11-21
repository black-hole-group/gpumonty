# Requirements:
# - pgcc, version >= 18.4
# - gcc, version < 7.0 (preferably 5.4.1)
# - cuda toolkit 9.2
# - cuda drivers
#
# makefile adapted from:
# https://hiltmon.com/blog/2013/07/03/a-simple-c-plus-plus-project-structure/
#

CC=pgcc
CCFLAGS=-fast -acc -ta=tesla:cc60,rdc -Minfo=, -O3 -Minform=warn
LDFLAGS=-Mcuda=rdc -lgsl -lgslcblas -Mcudalib=curand

NVCC=nvcc
NVCCFLAGS=-rdc=true -O3 -arch=sm_60

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
