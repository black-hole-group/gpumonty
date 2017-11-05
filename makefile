# requires an openmp-enabled version of gcc
#
CUDAC = nvcc
CUDAC_FLAGS = -Wno-deprecated-gpu-targets -O2
CC = gcc-5
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

SRCS = grmonty.c compton.c init_geometry.c tetrads.c geodesics.c \
radiation.c jnu_mixed.c hotcross.c track_super_photon.c \
scatter_super_photon.c harm_model.c harm_utils.c init_harm_data.c

OBJS = grmonty.o compton.o init_geometry.o tetrads.o geodesics.o \
radiation.o jnu_mixed.o hotcross.o track_super_photon.o \
scatter_super_photon.o harm_model.o harm_utils.o init_harm_data.o

INCS = decs.h constants.h harm_model.h

grmonty : $(OBJS) $(INCS) makefile
	$(CC) $(CFLAGS) -o grmonty $(OBJS) $(LDFLAGS)
	# $(CUDAC) $(ALL_CUDACFLAGS) $(ALL_CFLAGS) -o grmonty $(OBJS) $(ALL_LDFLAGS)

$(OBJS) : $(INCS) makefile

clean:
	/bin/rm *.o
