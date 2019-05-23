
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

#define SIZE 1000

__global__
void allocateMemory(float *d_mat_A, float *d_mat_B)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < SIZE && col < SIZE)
	{
		d_mat_A[row * SIZE + col] = 5.5;
		d_mat_B[row * SIZE + col] = 5.5;
	}
}

__global__
void matrixMultiplication(float *d_mat_A, float *d_mat_B, float *d_mat_C)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < SIZE && col < SIZE)
	{
		for (int i = 0; i < SIZE; i++)
		{
			d_mat_C[row * SIZE + col] += d_mat_A[col * SIZE + i] * d_mat_B[i * SIZE + col];
		}
	}
}

int main()
{
	float *h_mat_C = new float[SIZE*SIZE];
	
	float *d_mat_A, *d_mat_B, *d_mat_C;

	cudaMalloc((void**)&d_mat_A, SIZE*SIZE * sizeof(float));
	cudaMalloc((void**)&d_mat_B, SIZE*SIZE * sizeof(float));
	cudaMalloc((void**)&d_mat_C, SIZE*SIZE * sizeof(float));

	dim3 threadsPerBlock(256, 4);
	dim3 blocksPerGrid(40, 2500);

	allocateMemory<<<blocksPerGrid, threadsPerBlock>>>(d_mat_A, d_mat_B);
	matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(d_mat_A, d_mat_B, d_mat_C);
	cudaDeviceSynchronize();

	cudaMemcpy(h_mat_C, d_mat_C, SIZE*SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			printf("%.8f \t", h_mat_C[i * SIZE + j]);
		}
		printf("\n");
	}

	cudaFree(d_mat_A); cudaFree(d_mat_B); cudaFree(d_mat_C);
	delete[] h_mat_C;

	return 0;
}
