#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "harris.h"

typedef unsigned char Pixel;
#define PGM_MAX 255
#define PPM_TYPE 1
#define PGM_TYPE 2


float kth_smallest(float a[], int n, int k)
{
	register int i,j,l,m;
	register float x, tmp;

	l=0 ; m=n-1 ;
	while (l<m) {
		x=a[k];
		i=l;
		j=m;
		do {
			while (a[i]<x) i++;
			while (x<a[j]) j--;
			if (i<=j) {
				/* swap a[i], a[j] */
				tmp=a[i];
				a[i]=a[j];
				a[j]=tmp;
				i++; j--;
			}
		} while (i<=j);
		if (j<k) l=i;
		if (k<i) m=j;
	}
	return a[k];
}

#define _ABS(x) ((x)>=0? (x) : -(x))


float root4(float a)
{
	float x0, x1 = 1;

	if(!a) return 0;

	if(a < 0) return sqrtf(-1.0F); // nanf(); // 0.0F/0.0F;    /* NaN */

	do{
		x0 = x1;
		x1 = (3 * x1 + a / (x1*x1*x1))*0.25F;
	} while (_ABS(x0 - x1) >= _ABS(x1) * (FLT_EPSILON * 10));

	return x1;
}

int *initGaussian(float s, int *kernsz)
{
	register int x;
	int n, n2, *kern;
	float Gn, G;

	/* kernel size is ceil(3*s)*2+1, which for e.g. s=1 yields a 7-tap filter. s=0.5 yields a 5-tap filter */
	*kernsz=n2=(int)ceil(3.0*s)*2+1;
	n=n2>>1; // n2/2

	kern=(int *)malloc(n2*sizeof(int));
	//cudaMallocManaged((void**)&kern,n2*sizeof(int));

	if(!kern){
		fprintf(stderr, "memory allocation request failed in initGaussian()\n");
		exit(1);
	};

	x=n; Gn=(float)(1.0/(s*sqrt(2.0*M_PI))*expf(-x*x/(2.0F*s*s)));
	for(x=-n; x<=n; ++x){
		G=(float)(1.0/(s*sqrt(2.0*M_PI))*expf(-x*x/(2.0F*s*s)));
		kern[x+n]=(int)(G/Gn + 0.5F); // notice that Gn<=G(x) for all x, hence the ratio is >=1
	}


	return kern;
}

/* Note: relminthresh is taken into account only when NEGATIVE! */
void *harris_init(int imw, int imh, int n1, float relminthresh)
{
	int npixels;
	struct harrisData *hd;


	//cudaMallocManaged((void**)&hd,sizeof(struct harrisData));
	hd=(struct harrisData *)malloc(sizeof(struct harrisData));

	/* allocate and fill in convolution mask */
	hd->Gmask=initGaussian(HARRIS_SIGMA, &(hd->Gmasksz));

	hd->width=imw;
	hd->height=imh;
	hd->n1=n1;

	hd->relminthr=(relminthresh<0.0F)? -relminthresh : HARRIS_RELATIVE_MIN;

	/* setup work arrays */
	npixels=imw*imh;
	//hd->ibuf=(int *)malloc(5*npixels*sizeof(int));
	//cudaMallocManaged((void**)&(hd->ibuf),5*npixels*sizeof(int) );
	cudaMalloc((void**)&(hd->ibuf),5*npixels*sizeof(int) );


	if(!hd->ibuf){
		fprintf(stderr, "[int] memory allocation request failed in harris_init()!\n");
		exit(1);
	}

	hd->gradx=hd->ibuf;
	hd->grady=hd->gradx+npixels;
	hd->gradx2=hd->grady+npixels;
	hd->grady2=hd->gradx2+npixels;
	hd->gradxy=hd->grady2+npixels;

	cudaMalloc((void**)&(hd->fbuf), (npixels+HARRIS_MAX_CORNERS)*sizeof(float));

	if(!hd->fbuf){
		fprintf(stderr, "[float] memory allocation request failed in harris_init()!\n");
		exit(1);
	}
	hd->cornerness=hd->fbuf;
	hd->selcornerness=hd->cornerness+npixels;
	return (void *)hd;
}

void harris_finish(void *hdata)
{
	struct harrisData *hd=(struct harrisData *)hdata;

	/* cleanup */

	cudaFree(hd->ibuf);
	cudaFree(hd->fbuf);
	free(hd->Gmask);
	hd->ibuf=hd->gradx=hd->grady=hd->gradx2=hd->grady2=hd->gradxy=NULL;
	hd->fbuf=hd->cornerness=hd->selcornerness=NULL;
	cudaFree(hd);
}

#define CORNER_BLOB_HLF_SZ 1
void harris_drawCorners(unsigned char *img, int width, int height, float (*cornrs)[2], int ncorn)
{
	register int i, j, k;
	int x, y;

	//printf("%d\n", ncorn);
	for(k=0; k<ncorn; ++k){
		x=(int)cornrs[k][0]; y=(int)cornrs[k][1];
		if (x<0 || x>width || y<0 || y>height) printf("weird point ");
		else{
			for(i=-CORNER_BLOB_HLF_SZ; i<=CORNER_BLOB_HLF_SZ; ++i)
				for(j=-CORNER_BLOB_HLF_SZ; j<=CORNER_BLOB_HLF_SZ; ++j)
					img[(y+j)*width+x+i]=255;
			img[y*width+x]=0;
		}
	}
}

float frthrt(float x)
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