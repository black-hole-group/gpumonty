#
# requires an openmp-enabled version of gcc
#
CC = gcc
CCFLAGS  = -Wall -Ofast -fopenmp
LDFLAGS = -lm -lgsl -lgslcblas 

CC_COMPILE = $(CC) $(CCFLAGS) -c
CC_LOAD = $(CC) $(CCFLAGS)

.c.o:
	$(CC_COMPILE) $*.c

EXE = grmonty
all: $(EXE)

SRCS = grmonty.c compton.c init_geometry.c tetrads.c geodesics.c \
radiation.c jnu_mixed.c hotcross.c track_super_photon.c \
scatter_super_photon.c harm_model.c harm_utils.c init_harm_data_gustavo.c
 
OBJS = grmonty.o compton.o init_geometry.o tetrads.o geodesics.o \
radiation.o jnu_mixed.o hotcross.o track_super_photon.o \
scatter_super_photon.o harm_model.o harm_utils.o init_harm_data_gustavo.o

INCS = decs.h constants.h harm_model.h

$(OBJS) : $(INCS) makefile

$(EXE) : $(OBJS) $(INCS) makefile 
	$(CC_LOAD) $(CFLAGS) $(OBJS) $(LDFLAGS) -o $(EXE)

clean:
	/bin/rm *.o $(EXE)

