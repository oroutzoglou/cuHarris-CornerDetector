NVCC=nvcc
GCC=gcc
CUDAFLAGS= -arch=sm_30
OPT= -g -G
RM=/bin/rm -f
all: harris

helper_functions.o: helper_functions.cu	
	${NVCC} ${OPT} $(CUDAFLAGS)   -std=c++11 -c helper_functions.cu

io.o: io.cu	
	${NVCC} ${OPT} $(CUDAFLAGS)   -std=c++11 -c io.cu


kernels.o: kernels.cu	
	${NVCC} ${OPT} $(CUDAFLAGS)   -std=c++11 -c kernels.cu


harris_findCorners.o: harris_findCorners.cu
	${NVCC} ${OPT} $(CUDAFLAGS)   -std=c++11 -c harris_findCorners.cu


main.o: main.cu 
	$(NVCC) ${OPT} $(CUDAFLAGS)   -std=c++11 -c main.cu


harris: main.o harris_findCorners.o kernels.o io.o helper_functions.o
	${NVCC} ${CUDAFLAGS} -o harris main.o harris_findCorners.o kernels.o io.o helper_functions.o
	${RM} *.o
	
clean:
	${RM} *.o harris