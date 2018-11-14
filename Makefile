# Requirements:
# - cuda toolkit
# - cuda drivers
#
# makefile adapted from:
# https://hiltmon.com/blog/2013/07/03/a-simple-c-plus-plus-project-structure/
#

NVCC=nvcc
DEBUGFLAGS=-Xcompiler -rdynamic -G
CCFLAGS=-O3 -arch=sm_61 -rdc=true
LDFLAGS=-lgsl -lgslcblas -lcurand

GRMONTY_BASEBUILD ?= .
EXE = grmonty
SRCDIR = src
BUILDDIR = $(GRMONTY_BASEBUILD)/build
TARGET = $(GRMONTY_BASEBUILD)/bin/$(EXE)

SRCS = $(shell find $(SRCDIR) -type f -name *.cu)
OBJS = $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SRCS:.cu=.o))
INCS = $(shell find $(SRCDIR) -type f -name *.h)

.PHONY: all
all: $(TARGET)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu $(INCS) Makefile
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(CCFLAGS) -c  $< -o $@

$(TARGET): $(OBJS) $(INCS) Makefile
	$(NVCC) $(CCFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

.PHONY: clean
clean:
	/bin/rm -f $(TARGET); /bin/rm -rf $(GRMONTY_BASEBUILD)/build
