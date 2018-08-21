#
# requires an openmp-enabled version of gcc
#
CC = gcc
CCFLAGS  = -Wall -O2 -fopenmp
LDFLAGS = -lm -lgsl -lgslcblas

EXE = grmonty
SRCDIR = src
BUILDDIR = build
BINDIR = bin


SRCS = grmonty.c compton.c init_geometry.c tetrads.c geodesics.c \
radiation.c jnu_mixed.c hotcross.c track_super_photon.c \
scatter_super_photon.c harm_model.c harm_utils.c init_iharm2dv3_data.c

OBJS = grmonty.o compton.o init_geometry.o tetrads.o geodesics.o \
radiation.o jnu_mixed.o hotcross.o track_super_photon.o \
scatter_super_photon.o harm_model.o harm_utils.o init_iharm2dv3_data.o

INCS = decs.h constants.h harm_model.h


$(BINDIR)/$(EXE): $(BUILDDIR)/$(OBJS) $(INCS) makefile
	$(CC) $(CFLAGS) $(BUILDDIR)/$(OBJS) -o $(BINDIR)/$(EXE) $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c $(INCS) makefile
	@mkdir -p $(BUILDDIR)
	$(CC) $(CCFLAGS) -c $@ $<

.PHONY: clean
clean:
	/bin/rm *.o $(EXE)
