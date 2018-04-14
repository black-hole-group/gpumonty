# requires an openmp-enabled version of gcc
#
CC = nvcc
CFLAGS = -O2 -Xcompiler -fopenmp
LDFLAGS = -lm -lgsl -lgslcblas -lgomp

SRCS = grmonty.c compton.c init_geometry.c tetrads.c \
 jnu_mixed.c hotcross.c  \
 harm_model.c harm_utils.c init_harm_data.c 
 
OBJS = grmonty.o compton.o init_geometry.o tetrads.o  \
jnu_mixed.o hotcross.o  \
 harm_model.o harm_utils.o init_harm_data.o device_query.o

INCS = decs.h constants.h harm_model.h 

all: device_query.o grmonty

grmonty : $(OBJS) $(INCS) makefile 
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

#$(OBJS) : $(INCS) makefile
%.o: %.c $(INCS) makefile

device_query.o: device_query.cu
	$(CC) -c -o $@ $^

clean:
	/bin/rm *.o grmonty

