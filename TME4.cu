#include <iostream>
#include <stdio.h>
#include <random>
#include <algorithm>  // for std::sort()
#include <iomanip>    // for std::setw and std::setfill



__global__ void mergeSmall_k(float A[], float B[], unsigned short len_A, unsigned short len_B, float res[]) {
	struct {unsigned short x; unsigned short y;} P; struct {unsigned short x; unsigned short y;} K;
	const unsigned short i = threadIdx.x;
	if (i> len_A + len_B) return;
	if (i > len_A) {
		 K.x = i - len_A; K.y  = len_A;				// Low point of the diagonal
		 P.x =  len_A; P.y = i - len_A ;				// High point of the diagonal
	}
	else {
		K.x = 0; K.y = i;
		P.x = i; P.y = 0;
	}
	unsigned short offset; 
	struct {short x; short y;} Q;
	// short ct = 0;
	while (true) {
		/*
 		ct += 1;
		if (ct>20){
			printf("Thread : %i Q : %i, %i\n", i, Q.x, Q.y);
			break;
		} 
		*/
		offset = abs(K.y - P.y) / 2;
		Q.x =  K.x + offset; Q.y = K.y - offset;
		if ((Q.y >= 0) && (Q.x <= len_B) && (Q.x == 0 || Q.y == len_A || A[Q.y] > B[Q.x - 1])) {
			if (Q.x == len_B || Q.y == 0 || A[Q.y - 1] <= B[Q.x]) {
				if (Q.y < len_A && (Q.x == len_B || A[Q.y] <= B[Q.x])) {			// Merge in M
					res[i] = A[Q.y];
				}
				else {
					res[i] = B[Q.x];
				}
				break;
			}
			else {
				K.x = Q.x + 1; K.y = Q.y - 1;
			}
		}
		else {
			P.x = Q.x - 1; P.y = Q.y + 1;
		}
	}
}


__global__ void mergeSmall_block(float A[], float B[], size_t len_A, size_t len_B, float res[]) {
	struct {size_t x; size_t y;} P; struct {size_t x; size_t y;} K;
	const unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i> len_A + len_B) return;
	if (i > len_A) {
		 K.x = i - len_A; K.y  = len_A;				// Low point of the diagonal
		 P.x =  len_A; P.y = i - len_A ;				// High point of the diagonal
	}
	else {
		K.x = 0; K.y = i;
		P.x = i; P.y = 0;
	}
	size_t offset; 
	struct {size_t x; size_t y;} Q;
	while (true) {
		offset = abs((long) (K.y - P.y)) / 2;
		Q.x =  K.x + offset; Q.y = K.y - offset;
		if ((Q.y >= 0) && (Q.x <= len_B) && (Q.x == 0 || Q.y == len_A || A[Q.y] > B[Q.x - 1])) {
			if (Q.x == len_B || Q.y == 0 || A[Q.y - 1] <= B[Q.x]) {
				if (Q.y < len_A && (Q.x == len_B || A[Q.y] <= B[Q.x])) {			// Merge in M
					res[i] = A[Q.y];
				}
				else {
					res[i] = B[Q.x];
				}
				break;
			}
			else {
				K.x = Q.x + 1; K.y = Q.y - 1;
			}
		}
		else {
			P.x = Q.x - 1; P.y = Q.y + 1;
		}
	}
}

void gen_array(float tab[], size_t n){
	static std::default_random_engine rng;
    static std::uniform_real_distribution<> dis(0, 1); 
	for (size_t i = 0; i < n; ++i){
		tab[i] = dis(rng);
	}
	std::sort(tab, &tab[n]);
}

void test_merge(){
	const float A[] = {0.0002963553172046218, 0.1473706042389482, 0.15539382661327628, 0.1754163353567817, 0.3092625774755152, 0.5017953561658001, 0.5726677838352224, 0.5956352206355804, 0.6587386949266499, 0.819062253484973};
	const float B[] = {0.06887463948656714, 0.08465096544054529, 0.08640150169802019, 0.23489044053546804, 0.425905812285034, 0.5566190101207799, 0.5798369416262247, 0.651695037804542};

	float *aGPU, *bGPU, *OutGPU, Out[18];
	cudaMalloc(&aGPU, 10 * sizeof(float));
    cudaMalloc(&bGPU, 8  * sizeof(float));
    cudaMalloc(&OutGPU, 18 * sizeof(float));
    cudaMemcpy(aGPU, A, 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bGPU, B, 8  * sizeof(float), cudaMemcpyHostToDevice);

    mergeSmall_block<<<20, 5>>>(aGPU, bGPU, 10, 8, OutGPU);

    cudaMemcpy(Out, OutGPU, 18 * sizeof(float), cudaMemcpyDeviceToHost);
     
	for (auto n: Out)
	    std::cout<<n<<" ";
	
	std::cout<<std::endl;

}

void benchmark(){
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
  	cudaEventCreate(&stop);
	float ms;
	std::cout<<"********* Execution Time for different array size *********\n";
	for (unsigned int sz = 10; sz < 100000001; sz *= 10){
		float *A_tst = new float[sz/2];
		float *B_tst = new float[sz/2];
		float *Agpu, *Bgpu, *OutGPU; 
		gen_array(A_tst, sz/2); gen_array(B_tst, sz/2);
		cudaMalloc(&Agpu, sz/2 * sizeof(float));
    	cudaMalloc(&Bgpu, sz/2 * sizeof(float));
    	cudaMalloc(&OutGPU, sz  * sizeof(float));
		cudaMemcpy(Agpu, A_tst, sz/2 * sizeof(float), cudaMemcpyHostToDevice);
    	cudaMemcpy(Bgpu, B_tst, sz/2 * sizeof(float), cudaMemcpyHostToDevice);
		delete A_tst; delete B_tst;

	    cudaEventRecord(start);
    	mergeSmall_block<<<sz / 128 + 1, 128>>>(Agpu, Bgpu, sz/2, sz/2, OutGPU);
	  	cudaDeviceSynchronize();
    	cudaEventRecord(stop);
  		cudaEventSynchronize(stop);
	  	ms = 0;
  		cudaEventElapsedTime(&ms, start, stop);
		std::cout<<"array sz : "<<std::setfill(' ') << std::setw(10)<<sz<<", time : "<<ms<<" ms\n";
	}

}

int main(){
	benchmark();

    return 1;

}
