# requires an openmp-enabled version of gcc

CUDAC = nvcc
CUDAC_FLAGS = -Wno-deprecated-gpu-targets -O2
CC = g++-5
CFLAGS = -Wall -O2 -fopenmp
LDFLAGS = -lm -lgsl -lgslcblas

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

ALL_CFLAGS :=
ALL_CFLAGS += $(ALL_CFLAGS)
ALL_CFLAGS += $(addprefix -Xcompiler ,$(CFLAGS))

ALL_CUDACFLAGS :=
ALL_CUDACFLAGS += $(ALL_CUDACFLAGS)
ALL_CUDACFLAGS += $(CUDAC_FLAGS)
ALL_CUDACFLAGS += -ccbin $(CC)

SRCS = grmonty.cu compton.cu init_geometry.cu tetrads.cu geodesics.cu \
radiation.cu jnu_mixed.cu hotcross.cu track_super_photon.cu \
scatter_super_photon.cu harm_model.cu harm_utils.cu init_harm_data.cu \
gpu_helpers.cu

OBJS = grmonty.o compton.o init_geometry.o tetrads.o geodesics.o \
radiation.o jnu_mixed.o hotcross.o track_super_photon.o \
scatter_super_photon.o harm_model.o harm_utils.o init_harm_data.o \
gpu_helpers.o

INCS = decs.h constants.h harm_model.h gpu_helpers.h

grmonty : $(OBJS) $(INCS) makefile
	$(CUDAC) $(ALL_CUDACFLAGS) $(ALL_CFLAGS) -o grmonty $(OBJS) $(ALL_LDFLAGS)

$(OBJS) : $(INCS) makefile
	$(CUDAC) $(ALL_CUDACFLAGS) $(ALL_CFLAGS) -c $(SRCS)

clean:
	/bin/rm *.o

run:
	./grmonty 50000 dump019 4.e19
