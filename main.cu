#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include<cuda_runtime.h>
#include<cuda.h>
#include<stdlib.h>
#include "harris.h"
#include <time.h>
#include <sys/time.h>


struct timeval time_all0,time_all1;


int main(int argc, char *argv[])
{
	printf("Number of corners to be found is %d\n", HARRIS_NCORNERS);

	int width, height;
	char *fin, *fout;
	unsigned char *img;
	float (*cornrs)[2];
	int ncorn;
	void *hp;
	unsigned char *frdesc=NULL;

	if(argc!=3){
		fprintf(stderr, "usage: %s <in pgm> <out pgm>\n", argv[0]);
		exit(1);
	}
	fin=argv[1];
	fout=argv[2];

	// get input
	pgmReadHeader(fin, &height, &width);
	img=(unsigned char *)malloc(width*height*sizeof(unsigned char));
	unsigned char *device_img;
	cudaMalloc((void**)&device_img,width*height*sizeof(unsigned char ));
	pgmReadBuffer(fin, img, &height, &width);

    // CPU & GPU allocations
	cornrs=(float (*)[2])malloc(HARRIS_NCORNERS*sizeof(float[2]));            // corners coordinates on CPU mem
	float (*device_cornrs)[2];
	cudaMalloc((void**)&device_cornrs,HARRIS_NCORNERS*sizeof(float[2]));      // corners coordinates on GPU mem
	float (*cornrs_big)[2];
	cudaMallocManaged((void**)&cornrs_big,width*height*sizeof(float[2]));
	hp=harris_init(width, height, HARRIS_NCORNERS, HARRIS_RELATIVE_MIN_AUX);
	
	// derivatives
	int *gradx2_b;
	gradx2_b=(int *)malloc(width*height*sizeof(int));
	int *device_gradx2_b;
	cudaMalloc((void**)&device_gradx2_b,width*height*sizeof(int));
	int *grady2_b;
	grady2_b=(int *)malloc(width*height*sizeof(int));
	int *device_grady2_b;
	cudaMalloc((void**)&device_grady2_b,width*height*sizeof(int));
	int *gradxy_b;
	gradxy_b=(int *)malloc(width*height*sizeof(int));
	int *device_gradxy_b;
	cudaMalloc((void**)&device_gradxy_b,width*height*sizeof(int));
	float *maximum;
	maximum=(float *)malloc(sizeof(float));
	float *device_maximum;
	cudaMalloc((void**)&device_maximum, sizeof(float));
	int *dd_mutex;
	cudaMalloc((void**)&dd_mutex, sizeof(int));
	float *selcornerness2;
	selcornerness2=(float *)malloc(width*height*sizeof(float));
	float *device_selcornerness2;
	cudaMalloc((void**)&device_selcornerness2,width*height*sizeof(float ));
	int *gradx2_a;
	gradx2_a=(int *)malloc(width*height*sizeof(int));
	int *grady2_a;
	grady2_a=(int *)malloc(width*height*sizeof(int));
	int *gradxy_a;
	gradxy_a=(int *)malloc(width*height*sizeof(int));
	int *device_gradx2_a;
	cudaMalloc((void**)&device_gradx2_a,width*height*sizeof(int));
	int *device_grady2_a;
	cudaMalloc((void**)&device_grady2_a,width*height*sizeof(int));
	int *device_gradxy_a;
	cudaMalloc((void**)&device_gradxy_a,width*height*sizeof(int));
	int *gradx_a;
	gradx_a=(int *)malloc(width*height*sizeof(int));
	int *grady_a;
	grady_a=(int *)malloc(width*height*sizeof(int));
	int *device_gradx_a;
	cudaMalloc((void**)&device_gradx_a,width*height*sizeof(int));
	int *device_grady_a;
	cudaMalloc((void**)&device_grady_a,width*height*sizeof(int));
	int *atom;
	cudaMallocManaged((void**)&atom,sizeof(int ));
	float *r_thd_gpu;
	cudaMallocManaged((void**)&r_thd_gpu,sizeof(float ));
	int *seirial;
	cudaMallocManaged((void**)&seirial,sizeof(int ));
	int *atom2;
	atom2=(int *)malloc(sizeof(int));
	int *device_atom2;
	cudaMalloc((void**)&device_atom2,sizeof(int ));

	// warm up the GPU
	dim3 block(32,16);
	dim3 grid(16,32);
	kernel_warmingup<<<1,1>>>();
	cudaDeviceSynchronize();
   
	// find image's corners N times
	int N;
	N=10;
	for(int i=0;i<N;i++){
		*atom2=0;
		*seirial=0;
		*atom=0;
		*r_thd_gpu=0;
		gettimeofday(&time_all0,NULL);
		ncorn=harris_findCorners(hp, img,device_img, cornrs,device_cornrs,cornrs_big,gradx2_b,device_gradx2_b,
		grady2_b,device_grady2_b,gradxy_b,device_gradxy_b,maximum,device_maximum,dd_mutex,selcornerness2,
		device_selcornerness2,grid,block,gradx2_a,device_gradx2_a,grady2_a,device_grady2_a,gradxy_a,
		device_gradxy_a,atom,r_thd_gpu,seirial,atom2,device_atom2,gradx_a,device_gradx_a,grady_a,device_gradx_a);
		gettimeofday(&time_all1,NULL);
		double time_all10 = (time_all1.tv_sec*1000000.0 + time_all1.tv_usec) - (time_all0.tv_sec*1000000.0 + time_all0.tv_usec);
		fprintf(stderr, "total GPU  time: %lf msecs\n",  (time_all10)/1000.0F);
	}

#ifdef OUTPUT_THE_CORNERS_NOT_THE_IMAGE
	printf("Got %d corners, stored in corners_log.txt\n", ncorn);
	cornlog=fopen("corners_log.txt", "w");
	for(i=0; i<ncorn; ++i)
		fprintf(cornlog,"%.4f %.4f\n", cornrs[i][0], cornrs[i][1]);
	fclose(cornlog);
#else
	harris_drawCorners(img, width, height, cornrs, ncorn);
	pgmWriteBuffer(img, height, width, fout);
#endif
    // free CPU & GPU memory
	cudaFree(device_img);
	free(selcornerness2);
	cudaFree(device_selcornerness2);
	cudaFree(dd_mutex);
	free(atom2);
	cudaFree(device_atom2);
	cudaFree(seirial);
	free(gradx2_a);
	free(grady2_a);
	free(gradxy_a);
	cudaFree(device_gradx2_a);
	cudaFree(device_grady2_a);
	cudaFree(device_gradxy_a);
	cudaFree(cornrs_big);
	free(img);
	cudaFree(device_gradx2_b);
	cudaFree(device_grady2_b);
	cudaFree(device_gradxy_b);
	free(gradx2_b);
	free(grady2_b);
	free(gradxy_b);
	free(cornrs);
	cudaFree(device_cornrs);
	free(frdesc);
	harris_finish(hp);
	return 0;
}
