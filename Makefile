EXEC   = grmonty

OPTIMIZE =  -O2  

DIR = ./src # source dir
CFILES = $(wildcard $(DIR)/*.c)
CPPFILES = $(wildcard $(DIR)/*.cpp)
CUDAFILES = $(wildcard $(DIR)/*.cu)

OBJS   = $(subst .c,.o,$(CFILES)) $(subst .cpp,.o,$(CPPFILES)) $(subst .cu,.o,$(CUDAFILES)) 

# compilers
CC	= gcc
CXX   = g++
NVCC	= nvcc

.SUFFIXES : .c .cpp .cu .o

#CUDA_INCLUDE = -I/usr/local/cuda/include/
#CUDA_LIBS = -L/usr/local/cuda/lib64 -lcuda -lcudart
#INCL   = -I./ $(HDF5_INCLUDE)
#NVINCL = $(INCL) $(CUDA_INCLUDE)

NVLIBS = $(CUDA_LIBS)
LIBS   = -lm -lgsl -lgslcblas -lgomp



#FLAGS = $(CUDA) $(PRECISION) $(OUTPUT)$(COOLING) 
CFLAGS 	  = -O2 -Xcompiler -fopenmp 
CXXFLAGS  = $(CFLAGS) 
#NVCCFLAGS = $(FLAGS) -m64 -arch=compute_60 -fmad=false -ccbin=$(CC)
#LDFLAGS	  = -m64


%.o:	%.c
		$(CC) $(CFLAGS)  $(INCL)  -c $< -o $@ 

%.o:	%.cpp
		$(CXX) $(CXXFLAGS)  $(INCL)  -c $< -o $@ 

%.o:	%.cu
		$(NVCC) $(NVCCFLAGS)  $(INCL)  -c $< -o $@ 


$(EXEC): $(OBJS) 
	 	 $(NVCC) $(CXXFLAGS) $(OBJS) $(LDFLAGS) $(LIBS) $(NVLIBS) -o $(EXEC) $(INCL)   

#$(OBJS): $(INCL) 

.PHONY : clean

clean:
	 rm -f $(OBJS) $(EXEC)

