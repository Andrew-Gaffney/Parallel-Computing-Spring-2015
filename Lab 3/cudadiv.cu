#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define MAX_RAND 100001
#define WARP_SIZE 32

int size;
int nThreads;


/*Function to round up number of threads to nearest power of 2,
so that algorithm works(kind of a cheap workaround, I know, 
and also terrible for performance)
*/
int NearestPowerOf2 (int n) {
	
	if (!n) return n; 
  
	int x = 1;
  	while(x < n)
    {
      x <<= 1;
    }
  	return x;
}

//Simple function to get user input, set up thread count and fill random array
int* setup(char n[]){
	int i;
	int *rand_num;
	time_t t;
	
	//get size of array
	sscanf(n, "%d", &size);
	
	nThreads = NearestPowerOf2(size);
	
	srand((unsigned) time(&t));
	
	
	rand_num = (int*)malloc(size * sizeof(int));

    
	for(i = 0; i < size; i++){
		rand_num[i] = random() % MAX_RAND;
		//printf("%d\n", i);
	}
	
	return rand_num;	
	
}

/*
After kernel execution, max of input array will be somewhere within index 0
and 31 of array. Kernel is modified so that all threads in each warp are executing the same branch at the same time so as to avoid branch divergence.
*/ 
__global__ void find_max(int *rand, int numThreads) {
	
	int temp;
	int index = threadIdx.x + (blockDim.x * blockIdx.x);
		

	while(numThreads > WARP_SIZE){
		int halfway = numThreads / 2;	
		if (index < halfway){
			temp = rand[ index + halfway ];
			if (temp > rand[ index ]) {
				rand[index] = temp;
			}
		}
		__syncthreads();


		numThreads = halfway;	
	}
}

int main(int argc, char *argv[]){ 
	
	int *result = setup(argv[1]);
	int *devResult;
	int numBlocks;
	int m = 0;
	int i;
	
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int tPerBlock = prop.maxThreadsPerBlock;
	
	//Below condition only occcurs if nThreads is less than tPerBlock
	if ((nThreads % tPerBlock) != 0)
		numBlocks = (nThreads/tPerBlock) + 1;
	else
		numBlocks = nThreads/tPerBlock;
		
	
	cudaMalloc((void**)&devResult, size * sizeof(int));
	
	cudaMemcpy(devResult, result, size * sizeof(int), cudaMemcpyHostToDevice);
	
	find_max<<<numBlocks, tPerBlock>>>(devResult, nThreads);
	
	cudaMemcpy(result, devResult, size * sizeof(int), cudaMemcpyDeviceToHost);
	

	for(i = 0; i < WARP_SIZE; i++) {
		if(result[i] > m) {
			m = result[i];
		}
	}
	
	printf("The max value in the array is: %d\n", m);
	
	
	//printf("Number of threads is: %d\n", nThreads);
	

	//printf("Num of blocks is: %d\n", numBlocks);
	//printf("%d\n", prop.maxThreadsPerBlock);
		
	
	free(result);
	cudaFree(devResult);
	
	exit(0);
}
	
	
