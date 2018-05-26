# requires an openmp-enabled version of gcc
#
CC = nvcc
CFLAGS = -O2 -Xcompiler -fopenmp 
LDFLAGS = -lm -lgsl -lgslcblas -lgomp
 
OBJS = grmonty.o compton.o init_geometry.o tetrads.o  \
jnu_mixed.o hotcross.o harm_model.o harm_utils.o  \
init_harm_data.o device_query.o kernel.o gen_photons.o \
get_globals.o 

INCS = host.h host-device.h constants.h harm_model.h 

#all: device_query.o grmonty.o grmonty

grmonty: $(OBJS) $(INCS) makefile 
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cpp $(INCS) makefile

%.o: %.cu
	$(CC) -c $< -o $@ 	

clean:
	/bin/rm *.o grmonty

