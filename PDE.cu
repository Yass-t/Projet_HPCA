/**************************************************************
This code is a part of a course on cuda taught by the author:
Lokman A. Abbas-Turki

Those who re-use this code should mention in their code 
the name of the author above.
***************************************************************/

#include <stdio.h>
#include <math.h>

#define EPS 0.0000001f
#define NTPB 256
#define NB 64
#define r 0.1f

typedef float MyTab[NB][NTPB];

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

// Has to be defined in the compilation in order to get the correct value 
// of the macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

/*************************************************************************/
/*                   Black-Sholes Formula                                */
/*************************************************************************/
/*One-Dimensional Normal Law. Cumulative distribution function. */
double NP(double x){
  const double p= 0.2316419;
  const double b1= 0.319381530;
  const double b2= -0.356563782;
  const double b3= 1.781477937;
  const double b4= -1.821255978;
  const double b5= 1.330274429;
  const double one_over_twopi= 0.39894228;  
  double t;

  if(x >= 0.0){
	t = 1.0 / ( 1.0 + p * x );
    return (1.0 - one_over_twopi * exp( -x * x / 2.0 ) * t * ( t *( t * 
		   ( t * ( t * b5 + b4 ) + b3 ) + b2 ) + b1 ));
  }else{/* x < 0 */
    t = 1.0 / ( 1.0 - p * x );
    return ( one_over_twopi * exp( -x * x / 2.0 ) * t * ( t *( t * ( t * 
		   ( t * b5 + b4 ) + b3 ) + b2 ) + b1 ));
  }
}


// Parallel cyclic reduction for implicit part
__device__ void PCR_d(float *sa, float *sd, float *sc, 
					  float *sy, int *sl, int n){

	int i, lL, d, tL, tR;
	float aL, dL, cL, yL;
	float aLp, dLp, cLp, yLp;

	d = (n/2+(n%2))*(threadIdx.x%2) + (int)threadIdx.x/2;

	tL = threadIdx.x - 1;
	if (tL < 0) tL = 0;
	tR = threadIdx.x + 1;
	if (tR >= n) tR = 0;
	
	for(i=0; i<(int)(logf((float)n)/logf(2.0f))+1; i++){
		lL = (int)sl[threadIdx.x];

		aL = sa[threadIdx.x];
		dL = sd[threadIdx.x];
		cL = sc[threadIdx.x];
		yL = sy[threadIdx.x];

		dLp = sd[tL];
		cLp = sc[tL];

		if(fabsf(aL) > EPS){
			aLp = sa[tL];
			yLp = sy[tL];
			dL -= aL*cL/dLp;
			yL -= aL*yLp/dLp;
			aL = -aL*aLp/dLp;
			cL = -cLp*cL/dLp;
		}
		
		cLp = sc[tR];
		if(fabsf(cLp) > EPS){
			aLp = sa[tR];
			dLp = sd[tR];
			yLp = sy[tR];
			dL -= cLp*aLp/dLp;
			yL -= cLp*yLp/dLp;
		}
		__syncthreads();

		if (i < (int)(logf((float)n)/logf(2.0f))){
			sa[d] = aL;
			sd[d] = dL;
			sc[d] = cL;
			sy[d] = yL;
			sl[d] = (int)lL;	
			__syncthreads();
		}
	}

	sy[(int)sl[threadIdx.x]] = yL / dL;
}

__device__ float sqr(float a){
	return a * a;
}

/////////////////////////////////////////////////////////////////////////////
// A bad solution that makes a lot of accesses to the global memory
/////////////////////////////////////////////////////////////////////////////
__global__ void PDE_diff_k1 (float dt, float dx, float dsig, float pmin, 
							 float pmax, float sigmin, MyTab *pt_GPU){
	
	__shared__ float input[NTPB];
	__shared__ float output[NTPB];
	input[threadIdx.x] = pt_GPU[0][blockDim.x][threadIdx.x];
	__syncthreads();

	float sig = sigmin + blockIdx.x * dsig;
	float mu = r - sqr(sig) / 2.f;
    float p_u = (sqr(sig) * dt) / (2.f * sqr(dx)) + (mu * dt) / (2.f * dx);
	float p_m = 1.f - sqr(sig) * dt / sqr(dx);
	float p_d = (sqr(sig) * dt) / (2.f * sqr(dx)) - (mu * dt) / (2.f * dx);

	if (threadIdx.x == NTPB - 1){
		output[threadIdx.x] = pmax;
	}
	else if (threadIdx.x == 0){
		output[threadIdx.x] = pmin;
	} else {
		output[threadIdx.x] = p_u * input[threadIdx.x + 1] +  p_m * input[threadIdx.x] +   p_d * input[threadIdx.x - 1];
	}

	pt_GPU[0][blockIdx.x][threadIdx.x] = output[threadIdx.x];


}



// Wrapper 
void PDE_diff (float dt, float dx, float dsig, float pmin, float pmax, 
			   float sigmin, int N, MyTab* CPUTab){

	float TimeExec;									// GPU timer instructions
	cudaEvent_t start, stop;						// GPU timer instructions
	testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	testCUDA(cudaEventRecord(start,0));				// GPU timer instructions

	MyTab *GPUTab;
	testCUDA(cudaMalloc(&GPUTab, sizeof(MyTab)));
	
	testCUDA(cudaMemcpy(GPUTab, CPUTab, sizeof(MyTab), cudaMemcpyHostToDevice));
	// Accessing 2*N times to the global memory
	for(int i=0; i<N; i++){
	   PDE_diff_k1<<<NB,NTPB>>>(dt, dx, dsig, pmin, pmax, sigmin, GPUTab);
	}

	testCUDA(cudaMemcpy(CPUTab, GPUTab, sizeof(MyTab), cudaMemcpyDeviceToHost));

	testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
	testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
	testCUDA(cudaEventElapsedTime(&TimeExec,		// GPU timer instructions
			 start, stop));							// GPU timer instructions
	testCUDA(cudaEventDestroy(start));				// GPU timer instructions
	testCUDA(cudaEventDestroy(stop));				// GPU timer instructions

	printf("GPU time execution for PDE diffusion: %f ms\n", TimeExec);

	testCUDA(cudaFree(GPUTab));	
}

///////////////////////////////////////////////////////////////////////////
// main function for a put option f(x) = max(0,K-x)
///////////////////////////////////////////////////////////////////////////
int main(void){

	float K = 100.0f;
	float T = 1.0f;
	int N = 10000;
	float dt = (float)T/N;
	float xmin = log(0.5f*K);
	float xmax = log(2.0f*K);
	float dx = (xmax-xmin)/NTPB;
	float pmin = 0.5f*K;
	float pmax = 0.0f;
	float sigmin = 0.1f;
	float sigmax = 0.5f;
	float dsig = (sigmax-sigmin)/NB;
	

	MyTab *pt_CPU;
	testCUDA(cudaHostAlloc(&pt_CPU, sizeof(MyTab), cudaHostAllocDefault));
	for(int i=0; i<NB; i++){
	   for(int j=0; j<NTPB; j++){
	      pt_CPU[0][i][j] = max(0.0, K-exp(xmin + dx*j));	
	   }	
	}

	PDE_diff(dt, dx, dsig, pmin, pmax, sigmin, N, pt_CPU);

    // S0 = 100 , sigma = 0.2
	printf(" %f, compare with %f\n",exp(-r*T)*pt_CPU[0][16][128],
		   K*(exp(-r*T)*NP(-(r-0.5*0.2*0.2)*sqrt(T)/0.2)-
		   NP(-(r+0.5*0.2*0.2)*sqrt(T)/0.2)));
	// S0 = 100 , sigma = 0.3
	printf(" %f, compare with %f\n",exp(-r*T)*pt_CPU[0][32][128],
		   K*(exp(-r*T)*NP(-(r-0.5*0.3*0.3)*sqrt(T)/0.3)-
		   NP(-(r+0.5*0.3*0.3)*sqrt(T)/0.3)));
	// S0 = 141.4214 , sigma = 0.3
	printf(" %f, compare with %f\n",exp(-r*T)*pt_CPU[0][32][192],
		   K*exp(-r*T)*NP(-(log(141.4214/K)+(r-0.5*0.3*0.3)*T)/(0.3*sqrt(T)))-
		   141.4214*NP(-(log(141.4214/K)+(r+0.5*0.3*0.3)*T)/(0.3*sqrt(T))));

	testCUDA(cudaFreeHost(pt_CPU));	
	return 0;
}