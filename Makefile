#
# Requirements:
# - pgcc, version >= 18.4
# - gcc, version < 7.0 (preferably 5.4.1)
# - cuda toolkit 9.2
# - cuda drivers
#

#The GPU Compute Capability (change this according to your GPU)
GPU_CC=cc60
CC=pgcc
CCFLAGS=-fast -acc -ta=tesla:$(GPU_CC),rdc -Minfo=all -O2 -Minform=warn
CCFLAGS_FORCUDALIBS=-fast -acc -ta=tesla:$(GPU_CC),nollvm,rdc -Minfo=all -O2 -Minform=warn
LDFLAGS=-Mcuda=rdc -lgsl -lgslcblas -Mcudalib=curand
NVCC =nvcc
NVCCFLAGS = -rdc=true

EXE=grmonty
INCS=decs.h constants.h harm_model.h gmath.h gpu_rng.h bessel.h
OBJS = grmonty.o compton.o init_geometry.o tetrads.o geodesics.o \
radiation.o jnu_mixed.o hotcross.o track_super_photon.o \
scatter_super_photon.o harm_model.o harm_utils.o init_iharm2dv3_data.o\
cpu_rng.o gpu_rng.o gmath.o bessel.o acc_print.o

all: $(EXE)

acc_print.o: acc_print.cu
	$(NVCC) $(NVCCFLAGS) -c $<

gpu_rng.o: gpu_rng.c Makefile $(INCS)
	$(CC) $(CCFLAGS_FORCUDALIBS) -c $<

%.o: %.c Makefile $(INCS)
	$(CC) $(CCFLAGS) -c $<

$(EXE) : $(OBJS) $(INCS) Makefile
	$(CC) $(CCFLAGS) $(OBJS) -o $(EXE) $(LDFLAGS)

.PHONY: clean
clean:
	/bin/rm -f *.o $(EXE)
