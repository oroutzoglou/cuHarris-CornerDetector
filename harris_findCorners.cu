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

int harris_findCorners(void *hdata, unsigned char *img,unsigned char *device_img, float (*corners)[2],float (*device_corners)[2],float (*cornrs_big)[2],
int *gradx2_b,int *device_gradx2_b,int *grady2_b,int *device_grady2_b,int *gradxy_b,int *device_gradxy_b,float *maximum,float *device_maximum,int *dd_mutex,
float *selcornerness2,float *device_selcornerness2,dim3 grid,dim3 block,int *gradx2_a,int *device_gradx2_a,int *grady2_a,int *device_grady2_a,int *gradxy_a,
int *device_gradxy_a,int *atom,float *r_thd_gpu,int *seirial,int *atom2,int *device_atom2,int *gradx_a,int *device_gradx_a,int *grady_a,int *device_grady_a)
{
	register int	x;
	int		width, height;
	float	r_thd;
	int		nmaxima;
	int		offwidth, offheight;
	int		uoff, voff;

	struct harrisData *hd=(struct harrisData *)hdata;


	int    *device_gradx, *device_grady, *device_gradx2, *device_grady2, *device_gradxy;
	float  *cornerness;
	device_gradx=hd->gradx; device_grady=hd->grady;
	device_gradx2=hd->gradx2; device_grady2=hd->grady2; device_gradxy=hd->gradxy;
	hd->img=img;
	cornerness=hd->cornerness;
	width=hd->width; height=hd->height; // retrieve img dimensions

	cudaMemcpy(device_img, img, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
	kernel_imgradient5_smo<<<grid,block>>>(device_img, width, height, device_gradx2, device_grady2, device_gradx, device_grady);
	cudaDeviceSynchronize();

	voff=uoff=2;
	offwidth=width - uoff;
	offheight=height - voff;

	kernel_Ix2y2xy<<<grid,block>>>(device_gradx,device_grady,device_gradx2,device_grady2,device_gradxy,width,height);
	cudaDeviceSynchronize();

	x=3;
	kernel_imgblurg_separable_1<<<grid,block>>>(device_gradx2,device_gradx2_a,device_grady2,device_grady2_a,device_gradxy,device_gradxy_a,width,height);
	cudaDeviceSynchronize();
	kernel_imgblurg_separable_2<<<grid,block>>>(device_gradx2_a,device_gradx2_b,device_gradx2,device_grady2_a,device_grady2_b,device_grady2,device_gradxy_a,device_gradxy_b,device_gradxy,width,height);
	cudaDeviceSynchronize();

	uoff+=x;
	voff+=x;
	offwidth-=x;
	offheight-=x;

	/* compute the ``cornerness'' of each pixel */
	kernel_cornerness<<<grid,block>>>(device_gradx2_b,device_grady2_b,device_gradxy_b,cornerness,width,height);
	cudaDeviceSynchronize();
	unsigned int N = width*height;
	kernel_find_max<<< 128,256 >>>(cornerness, device_maximum, dd_mutex, N);
	cudaDeviceSynchronize();

#if 0 // 5x5
#else // 3x3
	uoff+=1;
	voff+=1;
	offwidth-=1;
	offheight-=1;
#endif
	nmaxima=0;
	kernel_memset<<<grid,block>>>(device_gradxy,width,height);
	cudaDeviceSynchronize();
	kernel_selcornerness<<<grid,block>>>(cornerness,device_selcornerness2,width,height,device_gradxy,device_maximum,seirial,hd->relminthr);
	cudaDeviceSynchronize();
	nmaxima=*seirial;      //number of corners found
	int zeros=width*height-nmaxima;


	if(nmaxima>hd->n1)
	{
		cudaMemcpy(selcornerness2, device_selcornerness2, width*height*sizeof(float), cudaMemcpyDeviceToHost);
		r_thd=kth_smallest(selcornerness2, width*height, zeros+ (nmaxima-hd->n1-1) );
	}
	else // not enough maxima, accept all
		r_thd=0.0;
	kernel_strong_corners<<<grid,block>>>(device_gradxy,cornerness,width,height);
	cudaDeviceSynchronize();
	cudaMemcpy(device_atom2, atom2, sizeof(int), cudaMemcpyHostToDevice);
	kernel_coordinates<<<grid,block>>>(cornerness,device_corners,device_gradxy,width,height,device_atom2);
	cudaDeviceSynchronize();
	cudaMemcpy(atom2, device_atom2, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(corners[0], device_corners[0], HARRIS_NCORNERS*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(corners[1], device_corners[1], HARRIS_NCORNERS*sizeof(float), cudaMemcpyDeviceToHost);
	int counter;
	counter=*atom2;
	return counter;
}







