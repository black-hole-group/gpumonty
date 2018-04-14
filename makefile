# requires an openmp-enabled version of gcc
#
CC = nvcc
CFLAGS = -O2 -Xcompiler -fopenmp
LDFLAGS = -lm -lgsl -lgslcblas -lgomp

SRCS = grmonty.cpp compton.cpp init_geometry.cpp tetrads.cpp \
 jnu_mixed.cpp hotcross.cpp  \
 harm_model.cpp harm_utils.cpp init_harm_data.cpp 
 
OBJS = grmonty.o compton.o init_geometry.o tetrads.o  \
jnu_mixed.o hotcross.o  \
 harm_model.o harm_utils.o init_harm_data.o device_query.o

INCS = decs.h constants.h harm_model.h 

all: device_query.o grmonty

grmonty : $(OBJS) $(INCS) makefile 
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

#$(OBJS) : $(INCS) makefile
%.o: %.cpp $(INCS) makefile

device_query.o: device_query.cu
	$(CC) -c -o $@ $^

clean:
	/bin/rm *.o grmonty

