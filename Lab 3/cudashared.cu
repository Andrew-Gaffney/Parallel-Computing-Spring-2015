#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define MAX_RAND 100001
#define WARP_SIZE 32

int size;
//int tpb;
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
	
	/*cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	tpb = prop.maxThreadsPerBlock;
	*/
	
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

#define T_PER_BLOCK 512//Ran on Cuda3
/* 
In the other versions of the program the threads per block was based on
the hardware that the kernel was being executed on. However, since the shared
memory array needs a constant value for its size, I could not figure out a way
to declare the threads per block based on the hardware.
*/

/*
After kernel execution, max of input array will be somewhere within index 0
and 31 of array. Kernel is modified so that all threads in each warp are executing the same branch at the same time so as to avoid branch divergence.
*/ 
__global__ void find_max(int *rand, int *maxArray) {
	
	int  temp;
	int numThreads = blockDim.x;
	__shared__ int  sharedMax[T_PER_BLOCK];
	int index = threadIdx.x + (blockDim.x * blockIdx.x);
	sharedMax[threadIdx.x] = rand[index];
	__syncthreads();
		

	while(numThreads > WARP_SIZE){
		int halfway = numThreads / 2;	
		if (threadIdx.x < halfway){
			temp = sharedMax[ threadIdx.x + halfway ];
			if (temp > sharedMax[ threadIdx.x ]) {
				sharedMax[threadIdx.x] = temp;
			}
		}
		__syncthreads();


		numThreads = halfway;	
	}
	maxArray[blockIdx.x] = sharedMax[0];
}

int main(int argc, char *argv[]){ 
	
	int *result = setup(argv[1]);
	int *hostMax;
	int *devMax;
	int *devResult;
	int numBlocks;
	
	int m = 0;
	int i; 
	
	//Below condition only occcurs if nThreads is less than tPerBlock
	if ((nThreads % T_PER_BLOCK) != 0)
		numBlocks = (nThreads/T_PER_BLOCK) + 1;
	else
		numBlocks = nThreads/T_PER_BLOCK;
	
	hostMax = (int*)malloc(numBlocks * sizeof(int));	
	
	cudaMalloc((void**)&devResult, size * sizeof(int));
	cudaMalloc((void**)&devMax, numBlocks * sizeof(int));
	
	cudaMemcpy(devResult, result, size * sizeof(int), cudaMemcpyHostToDevice);
	
	find_max<<<numBlocks, T_PER_BLOCK>>>(devResult, devMax);
	cudaMemcpy(hostMax, devMax, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	
	
	//Now that each threadblock has calculated its own max value, calculate
	//actual max value.
	m = hostMax[0];
	for(i = 0; i < numBlocks; i++) {
		if(hostMax[i] > m) {
			m = hostMax[i];
		}
	}
	
	printf("The max value in the array is: %d\n", m);
	
	
	//printf("Number of threads is: %d\n", nThreads);
	

	//printf("Num of blocks is: %d\n", numBlocks);
	//printf("%d\n", prop.maxThreadsPerBlock);
		
	
	free(result);
	free(hostMax);
	cudaFree(devResult);
	cudaFree(devMax);
	
	exit(0);
}
	
	
