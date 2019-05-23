#include<iostream>
#include<cuda_runtime.h>

#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1

__global__ void hello()
{
	printf("\n Hello World ! I am thread in block %d ", blockIdx.x);
}

int main() 
{
	hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();
	cudaDeviceSynchronize();
	printf("\n Thats all !");
	getchar();
	return 0;
}