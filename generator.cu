#include <emmintrin.h>
#include <sys/time.h>
#include <stdio.h>

const long N = 1000000; 
struct timeval start, end;


void starttime() {
  gettimeofday( &start, 0 );
}

void endtime(const char* c) {
   gettimeofday( &end, 0 );
   double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
   printf("%s: %f ms\n", c, elapsed); 
}

void init(const char* c) {
  printf("***************** %s **********************\n", c); 
  printf("Running %s...\n", c);
  starttime();
}

void finish(int a, long N, const char* c) {
	endtime(c);
	printf("Done.\n");
	printf("\nThere are %ld Prime numbers between 1 and %ld.", a, N);
	printf("***************************************************\n");
}

int normal(int a, long N)
{
    long low = 2, high = N, i, check;
	// printf("Prime numbers between 1 and %d are: ",high);
	while (low < high)
	{
		check = 0;
		for(i = 2; i <= low/2; ++i)
		{
		if(low % i == 0)
		{
			check = 1;
			break;
		}
		}
		if (check == 0)
			++a;
		//printf("%d ", low);
		++low;
   }
   return a;
}                                                                                                                                                                                                       

// GPU function to countprime numbers
// Every thread on every core runs this function
__global__ void gpu_prime(int* a, long N) {
   // One element per thread on each core
   // blockIdx.x = Core #
   // blockDim.x = Threads per core
   // threadIdx.x = Thread #
   // The formula below makes sure the value of element 
   // is different on every thread on every core
   long element = blockIdx.x*blockDim.x + threadIdx.x;
   
   // If there is not an event split, some threads will be 
   // out of bounds
   // We just let those do nothing
   // The rest count the prime numbers 
   
	if (element <= N && element >= 2) {
		int check = 0;	
		for(int i = 2; i <= element/2; ++i) {
        		if(element  % i == 0) {
        		check = 1;
        		break;
        		}
        	}
		if (check == 0){
	                atomicAdd(a,1);	
		}
	}
}

void gpu(int* a, long N) {
   int threadsPerCore = 512; // This can vary, up to 1024
   long numCores = N / threadsPerCore + 1; 

  
   // Memory must be on the graphics card 
   int* gpuA;
   cudaMalloc(&gpuA, sizeof(int)); // Allocate enough memory on the GPU
   
   cudaMemcpy(gpuA, a, sizeof(int), cudaMemcpyHostToDevice); 
   gpu_prime<<<numCores, threadsPerCore>>>(gpuA, N);
   cudaMemcpy(a, gpuA, sizeof(int), cudaMemcpyDeviceToHost); 
   cudaFree(&gpuA); // Free the memory on the GPU
}
                                                                                                                                                                                               
 

int main()                                                                                                                                                                                  
{

	int a = 1;
	
	// Test 1: Sequential For Loop
	init ("Normal");
	a = normal(a, N); 
	finish(a, N, "Normal"); 
	// Test 2: GPU
	a = 1;
	init("GPU");
	gpu(&a, N);  
	finish(a, N, "GPU");
  
	return 0;
}

