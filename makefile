#
# requires pgcc and nvcc
#
CC=pgcc
CCFLAGS=-fast -acc -ta=tesla,cc60,nollvm,rdc -Minfo=all -O2 -Minform=warn
LDFLAGS=-Mcuda -lgsl -lgslcblas -Mcudalib=curand
# CUDAC=nvcc
# CUDAFLAGS=

INCS=decs.h constants.h harm_model.h gmath.h gpu_rng.h
EXE=grmonty

all: $(EXE)

%.o: %.c makefile $(INCS)
	$(CC) $(CCFLAGS) -c $< $(LDFLAGS)

# %.o: %.cu makefile $(INCS)
# 	$(CUDAC) $(CUDAFLAGS) -c $<


OBJS = grmonty.o compton.o init_geometry.o tetrads.o geodesics.o \
radiation.o jnu_mixed.o hotcross.o track_super_photon.o \
scatter_super_photon.o harm_model.o harm_utils.o init_iharm2dv3_data.o\
cpu_rng.o gpu_rng.o gmath.o

$(EXE) : $(OBJS) $(INCS) makefile
	$(CC) $(CCFLAGS) $(OBJS) -o $(EXE) $(LDFLAGS)

.PHONY: clean
clean:
	/bin/rm -f *.o grmonty
