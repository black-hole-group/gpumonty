#
# requires an openmp-enabled version of gcc
#
# makefile adapted from https://hiltmon.com/blog/2013/07/03/a-simple-c-plus-plus-project-structure/
#
CC = gcc
CCFLAGS  = -Wall -O2 -fopenmp
LDFLAGS = -lm -lgsl -lgslcblas

EXE = grmonty
SRCDIR = src
BUILDDIR = build
TARGET = bin/$(EXE)

EXCLUDE=init_harm_data.c
SRCS = $(shell find $(SRCDIR) -type f -name *.c | grep -v $(EXCLUDE))
OBJS = $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SRCS:.c=.o))
INCS = $(shell find $(SRCDIR) -type f -name *.h)


$(TARGET): $(OBJS) $(INCS) makefile
	$(CC) $(CCFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c $(INCS) makefile
	@mkdir -p $(BUILDDIR)
	$(CC) $(CCFLAGS) -c  $< -o $@

.PHONY: clean
clean:
	/bin/rm -f $(TARGET); /bin/rm -rf ./build
