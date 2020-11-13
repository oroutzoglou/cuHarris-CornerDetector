#define USE_SQUARE_KERNEL_DERIVATIVES
#define USE_SQUARE_KERNEL_BLUR
#define DO_NOT_USE_BORDER_PIXELS
//#define OUTPUT_THE_CORNERS_NOT_THE_IMAGE
#ifdef DO_FIXED_POINT_MODEL
#define HARAUX 16.0F //reduce harris K bits (see HAR1, HAR2). Now: 16 => truncate 4 bits. K'=16*K=0.64
#define HAR1 (1/HARAUX) //harris, for the truncation of derivatives
#define HAR2 fixwidth((0.04F*HARAUX),1,3,'s') // [7 bits: mean error~0.001] [3 bits: mean error~0.003]
#define CORN_INTG_BITS 22
#define CORN_FRAC_BITS -5 //negative values denote the intg LSB to truncate in FPGA (+ all frac bits)
#endif

#define HARRIS_NCORNERS           20000    // define here the number of corners you need to find
#define HARRIS_MAX_CORNERS        20000
#define HARRIS_KAPPA              0.04F
#define HARRIS_SIGMA              1.0F //1.5F //0.7F
#define HARRIS_RELATIVE_MIN       0.035F //0.000005F //0.00001F //<--USE THIS, NOT THE AUX
#define HARRIS_RELATIVE_MIN_AUX   1 //-1e-05F
#define ONE_THIRD_GPU   0.33333333333333333333F /* 1/3 */
#define ONE_FOURTH_GPU  0.25F                   /* 1/4 */
#define ONE_SIXTH_GPU   0.16666666666666666666F /* 1/6 */
#define allnorm_kernel (1.0F/26.0F)*(1.0F/10.0F)
#define norm_kernel 1.0/51076
#define NUM_PASSES                2 //1
#define FRTHRT(x) frthrt(x) // faster version
#ifndef M_PI
#define M_PI 3.14159265358979323846 /* pi */
#endif /* M_PI */
#define PGM_MAX 255
#define PPM_TYPE 1
#define PGM_TYPE 2

typedef unsigned char Pixel;

struct harrisData{
	int width, height; /* image width & height */
	int *Gmask, Gmasksz;
	int n1; /* n1: max number of corners to be detected in an image */
	float relminthr; /* min relative threshold for declaring the cornerness of a point as a local maximum */

	/* work arrays */
	int *ibuf, *gradx, *grady, *gradx2, *grady2, *gradxy;
	float *fbuf, *cornerness, *selcornerness;

	/* pointer to pixel data, used by DCT descriptor only */
	unsigned char *img;
};

// CUDA kernels
__global__ void kernel_warmingup();
__global__ void kernel_coordinates(float* cornerness, float (*corners)[2],int *gradxy,int width,int height,int *atom2);
__global__ void kernel_strong_corners(int *gradxy,float *cornerness,int width,int height);
__global__ void kernel_memset(int *gradxy,int width,int height);
__global__ void kernel_imgblurg_separable_1(int *gradx2,int *result,int *grady2,int *result1,int *gradxy,int *result2,int width,int height);
__global__ void kernel_imgblurg_separable_2(int *gradx2,int *result,int *input,int *grady2,int *result1,int *input1,int *gradxy,int *result2,int *input2,int width,int height);
__global__ void kernel_find_max(float *array, float *max, int *mutex, unsigned int n);
__global__ void kernel_selcornerness(float *cornerness,float *selcornerness,int width, int height,int *gradxy,float *maximum,int *seirial,float RELMINTHR);
__global__ void kernel_cornerness(int *gradx2_b,int *grady2_b,int *gradxy_b,float *cornerness,int width,int height);
__global__ void kernel_Ix2y2xy(int *gradx,int *grady,int *gradx2,int *grady2,int *gradxy,int width,int height);
__global__ void kernel_imgradient5_smo(unsigned char *img,int width,int height, int *gradx2,int *grady2, int *gradx, int *grady);

// CPU functions
void *harris_init(int imw, int imh, int n1, float relminthresh);
void harris_finish(void *hdata);
void error(char *message);
void pgmReadHeader(char *name, int *rows, int *columns);
void ppmReadHeader(char *name, int *rows, int *columns);
void int2Char(int num, FILE *f);
void pgmWriteBuffer(Pixel *img, int rows, int columns, char *name);
void ppmWriteBuffer(Pixel *img, int rows, int columns, char *name);
void harris_drawCorners(unsigned char *img, int width, int height, float (*cornrs)[2], int ncorn);

int isWhiteSpace(int num);
int char2Int(FILE *f);
int pgmReadBuffer(char *name, Pixel *img, int *rows, int *columns);
int ppmReadBuffer(char *name, Pixel *img, int *rows, int *columns);
int ppmOrpgmReadBuffer(char *name, Pixel *img, int *rows, int *columns);
int *initGaussian(float s, int *kernsz);
int harris_findCorners(void *hdata, unsigned char *img,unsigned char *device_img, float (*corners)[2],float (*device_corners)[2],float (*cornrs_big)[2], 
int *gradx2_b,int *device_gradx2_b,int *grady2_b,int *device_grady2_b,int *gradxy_b,int *device_gradxy_b,float *maximum,float *device_maximum,int *dd_mutex, 
float *selcornerness2,float *device_selcornerness2,dim3 grid,dim3 block,int *gradx2_a,int *device_gradx2_a,int *grady2_a,int *device_grady2_a,int *gradxy_a, 
int *device_gradxy_a,int *atom,float *r_thd_gpu,int *seirial,int *atom2,int *device_atom2,int *gradx_a,int *device_gradx_a,int *grady_a,int *device_grady_a);

float frthrt(float x);
float root4(float x);
float kth_smallest(float a[], int n, int k);

