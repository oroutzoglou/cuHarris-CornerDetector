#define allnorm_kernel (1.0F/26.0F)*(1.0F/10.0F)
#define norm_kernel 1.0/51076
#define ONE_THIRD_GPU   0.33333333333333333333F /* 1/3 */
#define ONE_FOURTH_GPU  0.25F                   /* 1/4 */
#define ONE_SIXTH_GPU   0.16666666666666666666F /* 1/6 */

__global__ void kernel_warmingup(){}

__global__ void kernel_strong_corners(int *gradxy,float *cornerness,int width,int height)
{
	//printf("%d ",*atom);
	int hh=threadIdx.x+blockIdx.x*blockDim.x;
	int gg=threadIdx.y+blockIdx.y*blockDim.y;
	int N=width;

	if( (hh>=6)&&(hh<(width-6))&&(gg>=6)&&(gg<(height-6))/*&&(gradxy[gg*N+hh])&&(cornerness[gg*N+hh]<28744.000000)*/ )
	{


		if(cornerness[gg*N+hh]<0)
			gradxy[gg*N+hh]=0;

	}
}

__global__ void kernel_memset(int *gradxy,int width,int height)
{
	int gg=threadIdx.x+blockIdx.x*blockDim.x;
	int hh=threadIdx.y+blockIdx.y*blockDim.y;
	if((hh<width)&&(gg<height))
	{
		gradxy[gg*width+hh]=0;
	}

}

__global__ void kernel_imgblurg_separable_1(int *gradx2,int *result,int *grady2,int *result1,int *gradxy,int *result2,int width,int height)
{
	int imageW=height;
	int gg=threadIdx.x+blockIdx.x*blockDim.x;
	int hh=threadIdx.y+blockIdx.y*blockDim.y;
	if((hh<width)&&(gg<height))
	{
		result[hh*imageW+gg]=1*gradx2[hh*imageW+gg-3]+12*gradx2[hh*imageW+gg-2]+55*gradx2[hh*imageW+gg-1]+90*gradx2[hh*imageW+gg]+55*gradx2[hh*imageW+gg+1]+
			12*gradx2[hh*imageW+gg+2]+1*gradx2[hh*imageW+gg+3];

		result1[hh*imageW+gg]=1*grady2[hh*imageW+gg-3]+12*grady2[hh*imageW+gg-2]+55*grady2[hh*imageW+gg-1]+90*grady2[hh*imageW+gg]+55*grady2[hh*imageW+gg+1]+
			12*grady2[hh*imageW+gg+2]+1*grady2[hh*imageW+gg+3];

		result2[hh*imageW+gg]=1*gradxy[hh*imageW+gg-3]+12*gradxy[hh*imageW+gg-2]+55*gradxy[hh*imageW+gg-1]+90*gradxy[hh*imageW+gg]+55*gradxy[hh*imageW+gg+1]+
			12*gradxy[hh*imageW+gg+2]+1*gradxy[hh*imageW+gg+3];


	}
}


__global__ void kernel_imgblurg_separable_2(int *gradx2,int *result,int *input,int *grady2,int *result1,int *input1,int *gradxy,int *result2,int *input2,int width,int height)
{
	int imageW=width;
	int gg=threadIdx.x+blockIdx.x*blockDim.x;
	int hh=threadIdx.y+blockIdx.y*blockDim.y;
	if((gg<width)&&(hh<height))
	{
		if((gg>=3)&&(hh<(height-3))&&(hh>=3)&&(gg<(width-3)))

		{

			result[hh*imageW+gg]=norm_kernel*(1*gradx2[(hh-3)*imageW+gg]+12*gradx2[(hh-2)*imageW+gg]+55*gradx2[(hh-1)*imageW+gg]+90*gradx2[hh*imageW+gg]+55*gradx2[(hh+1)*imageW+gg]+
					12*gradx2[(hh+2)*imageW+gg]+1*gradx2[(hh+3)*imageW+gg]);

			result1[hh*imageW+gg]=norm_kernel*(1*grady2[(hh-3)*imageW+gg]+12*grady2[(hh-2)*imageW+gg]+55*grady2[(hh-1)*imageW+gg]+90*grady2[hh*imageW+gg]+55*grady2[(hh+1)*imageW+gg]+
					12*grady2[(hh+2)*imageW+gg]+1*grady2[(hh+3)*imageW+gg]);

			result2[hh*imageW+gg]=norm_kernel*(1*gradxy[(hh-3)*imageW+gg]+12*gradxy[(hh-2)*imageW+gg]+55*gradxy[(hh-1)*imageW+gg]+90*gradxy[hh*imageW+gg]+55*gradxy[(hh+1)*imageW+gg]+
					12*gradxy[(hh+2)*imageW+gg]+1*gradxy[(hh+3)*imageW+gg]);

		}
		else{

			result[hh*imageW+gg]= input[hh*imageW+gg];
			result1[hh*imageW+gg]= input1[hh*imageW+gg];
			result2[hh*imageW+gg]= input2[hh*imageW+gg];

		}

	}
}

__global__ void kernel_find_max(float *array, float *max, int *mutex, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ float cache[256];


	float temp = -1.0;
	while(index + offset < n){
		temp = fmaxf(temp, array[index + offset]);

		offset += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();


	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0){
		while(atomicCAS(mutex,0,1) != 0);  //lock
		*max = fmaxf(*max, cache[0]);
		atomicExch(mutex, 0);  //unlock
	}
}






__global__ void kernel_selcornerness(float *cornerness,float *selcornerness,int width, int height,int *gradxy,float *maximum,int *seirial,float RELMINTHR)
{
	int hh=threadIdx.x+blockIdx.x*blockDim.x;
	int gg=threadIdx.y+blockIdx.y*blockDim.y;
	int N=width;
	float d;
	if((hh>=6)&&(hh<(width-6))&&(gg>=6)&&(gg<(height-6)))
	{
		d=cornerness[gg*N+hh];
		if(d>=((*maximum)*RELMINTHR))
		{
			if ( (d>cornerness[(gg)*N+(hh+1)])  &&(d>=cornerness[(gg)*N+(hh-1)])&&(d>=cornerness[(gg+1)*N+(hh+1)])
					&&(d>=cornerness[(gg+1)*N+(hh-1)])&&(d>cornerness[(gg-1)*N+(hh+1)] )  && (d>cornerness[(gg-1)*N+(hh-1)]) &&(d>=cornerness[(gg+1)*N+(hh)]) &&(d>cornerness[(gg-1)*N+(hh)])   )
			{
				atomicAdd(seirial,1);

				gradxy[gg*N+hh]=1;
			}


		}

	}

}


__global__ void kernel_cornerness(int *gradx2_b,int *grady2_b,int *gradxy_b,float *cornerness,int width,int height)
{
	int hh=threadIdx.x+blockIdx.x*blockDim.x;
	int gg=threadIdx.y+blockIdx.y*blockDim.y;
	int N=width;

	if((hh>=5)&&(hh<(width-5))&&(gg>=5)&&(gg<(height-5)))
	{

		int det,trace;
		float r;
		int gxx,gyy,gxy;

		gxx=gradx2_b[gg*N+hh];
		gyy=grady2_b[gg*N+hh];
		gxy=gradxy_b[gg*N+hh];

		det=gxx*gyy - gxy*gxy;
		trace=gxx + gyy;



		r=det - 0.04*trace*trace;

		if(r<0.0F) r=0.0F;



		cornerness[gg*N+hh]=r;

	}
}



__global__ void kernel_Ix2y2xy(int *gradx,int *grady,int *gradx2,int *grady2,int *gradxy,int width,int height)
{
	int hh=threadIdx.x+blockIdx.x*blockDim.x;
	int gg=threadIdx.y+blockIdx.y*blockDim.y;
	int N=width;
	if((hh>=0)&&(hh<width)&&(gg>=0)&&(gg<height))
	{
		gradx2[gg*N+hh]= gradx[gg*N+hh]* gradx[gg*N+hh];
		grady2[gg*N+hh]= grady[gg*N+hh]* grady[gg*N+hh];
		gradxy[gg*N+hh]= gradx[gg*N+hh]* grady[gg*N+hh];




	}
}

__global__ void kernel_imgradient5_smo(unsigned char *img,int width,int height, int *gradx2,int *grady2, int *gradx, int *grady)
{
	int hh=threadIdx.x+blockIdx.x*blockDim.x;
	int gg=threadIdx.y+blockIdx.y*blockDim.y;
	if((hh>=2)&&(hh<(width-2))&&(gg>=2)&&(gg<(height-2)))
	{
		int N=width;
		gradx[gg*N+hh]=allnorm_kernel*(36*(img[gg*N+hh+1]-img[gg*N+hh-1]) +
				18*(img[(gg+1)*N+hh+1]+img[(gg-1)*N+hh+1]-img[(gg-1)*N+hh-1]-img[(gg+1)*N+hh-1]) +
				12*(img[(gg*N+hh+2)]-img[gg*N+hh-2]) +
				6*(img[(gg+1)*N+hh+2]+img[(gg-1)*N+hh+2]-img[(gg+1)*N+hh-2]-img[(gg-1)*N+hh-2]) +
				3*(img[(gg+2)*N+hh+1]+img[(gg-2)*N+hh+1]-img[(gg+2)*N+hh-1]-img[(gg-2)*N+hh-1]) +
				1*(img[(gg+2)*N+hh+2]+img[(gg-2)*N+hh+2]-img[(gg-2)*N+hh-2]-img[(gg+2)*N+hh-2]));

		grady[gg*N+hh] = allnorm_kernel*(36*(img[(gg+1)*N+hh]-img[(gg-1)*N+hh]) +
				18*(img[(gg+1)*N+hh+1]+img[(gg+1)*N+hh-1]-img[(gg-1)*N+hh+1]-img[(gg-1)*N+hh-1]) +
				12*(img[(gg+2)*N+hh]-img[(gg-2)*N+hh]) +
				6*(img[(gg+2)*N+hh+1]+img[(gg+2)*N+hh-1]-img[(gg-2)*N+hh+1]-img[(gg-2)*N+hh-1]) +
				3*(img[(gg+1)*N+hh+2]+img[(gg+1)*N+hh-2]-img[(gg-1)*N+hh+2]-img[(gg-1)*N+hh-2]) +
				1*(img[(gg+2)*N+hh+2]+img[(gg+2)*N+hh-2]-img[(gg-2)*N+hh+2]-img[(gg-2)*N+hh-2]));

	}
}

__device__ float kati(float x)
{
	float xhalf;
	int i;

	/* compute inverse square root */
	xhalf=0.5f*x;
	i=*(int*)&x;
	i=0x5f375a86 - (i>>1); // hidden initial guess, fast - LOMONT
	x=*(float*)&i;
	x=x*(1.5f-xhalf*x*x);
	x=x*(1.5f-xhalf*x*x); // add this in for added precision, or many more...

	/* compute fourth root as the inverse square root of the inverse square root */
	xhalf=0.5f*x;
	i=*(int*)&x;
	i=0x5f375a86 - (i>>1); // hidden initial guess, fast - LOMONT
	x=*(float*)&i;
	x=x*(1.5f-xhalf*x*x);
	x=x*(1.5f-xhalf*x*x); // add this in for added precision, or many more...
	return x;
}



__global__ void kernel_coordinates(float* cornerness, float (*corners)[2],int *gradxy,int width,int height,int *atom2)
{
	int hh=threadIdx.x+blockIdx.x*blockDim.x;
	int gg=threadIdx.y+blockIdx.y*blockDim.y;
	int N=width;
	float spp, spc, spn, scp, scc, scn, snp, snc, snn;
	float Pxx, Pxy, Pyy, Px, Py, ucorr, vcorr, detf;

	if((hh>=6)&&(hh<(width-6))&&(gg>=6)&&(gg<(height-6)))
	{
		if(gradxy[gg*N+hh])
		{
			spp=kati(cornerness[(gg-1)*N+(hh-1)]);
			spc=kati(cornerness[(gg-1)*N+(hh)]);
			spn=kati(cornerness[(gg-1)*N+(hh+1)]);
			scp=kati(cornerness[(gg)*N+(hh-1)]);
			scc=kati(cornerness[(gg)*N+(hh)]);
			scn=kati(cornerness[(gg)*N+(hh+1)]);
			snp=kati(cornerness[(gg+1)*N+(hh-1)]);
			snc=kati(cornerness[(gg+1)*N+(hh)]);
			snn=kati(cornerness[(gg+1)*N+(hh+1)]);
			Pxx=(spp + scp + snp  -2.0F*(spc + scc + snc) +  spn + scn + snn)*ONE_THIRD_GPU;
			Pxy=(spp - spn - snp + snn)*ONE_FOURTH_GPU;
			Pyy=        (spp +  spc +  spn
					-2.0F*(scp +  scc +  scn)
					+      snp +  snc +  snn)*ONE_THIRD_GPU;
			Px=(- spp - scp - snp + spn + scn + snn)*ONE_SIXTH_GPU;
			Py=(- spp - spc - spn + snp + snc + snn)*ONE_SIXTH_GPU;
			detf=Pxy*Pxy - Pxx*Pyy;
			if(detf>=1E-12F || detf<=-1E-12F){ // nonzero determinant
				//   calculate sub-pixel corrections to the corner position
				ucorr=(Pyy*Px - Pxy*Py)/detf;
				vcorr=(Pxx*Py - Pxy*Px)/detf;

				//   pull the corrections inside the pixel
				// printf("a=%lf ",ucorr );
				if(ucorr>0.5F) ucorr=0.5F; else if(ucorr<-0.5F) ucorr=-0.5F;
				if(vcorr>0.5F) vcorr=0.5F; else if(vcorr<-0.5F) vcorr=-0.5F;
				// printf("xx");
			}
			else
			{
				ucorr=vcorr=0.0F;
			}
			int x=atomicAdd(atom2,1);
			//cornrs_big[gg*N+hh][0]=/*u0*/hh+ucorr;
			//cornrs_big[gg*N+hh][1]=/*v0*/gg+vcorr;
			//printf("%lf ",ucorr);
			corners[x][0]=hh+ucorr;
			corners[x][1]=gg+vcorr;
		}
	}
}
