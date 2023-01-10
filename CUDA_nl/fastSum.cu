//Source https://github.com/mattdean1/cuda

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdint.h"
#include <device_functions.h>

#include "fastSum.cuh"

#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)



__global__ void prescan_arbitrary(uint32_t* output, uint32_t* input, uint32_t n, uint32_t powerOfTwo, uint32_t l)
{
	extern __shared__ uint32_t temp[];// allocated on invocation
	uint32_t threadID = threadIdx.x;

	uint32_t ai = threadID;
	uint32_t bi = threadID + (n / 2);
	uint32_t bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	uint32_t bankOffsetB = CONFLICT_FREE_OFFSET(bi);


	if (threadID < n) {
		temp[ai + bankOffsetA] = input[ai + l * n];
		temp[bi + bankOffsetB] = input[bi + l * n];
	}
	else {
		temp[ai + bankOffsetA] = 0;
		temp[bi + bankOffsetB] = 0;
	}


	uint32_t offset = 1;
	for (uint32_t d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			uint32_t ai = offset * (2 * threadID + 1) - 1;
			uint32_t bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) {
		temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0; // clear the last element
	}

	for (uint32_t d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			uint32_t ai = offset * (2 * threadID + 1) - 1;
			uint32_t bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			uint32_t t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadID < n) {
		output[ai] = temp[ai + bankOffsetA];
		output[bi] = temp[bi + bankOffsetB];
	}
}

__global__ void prescan_large(uint32_t* output, uint32_t* input, uint32_t n, uint32_t* sums, uint32_t l) {
	extern __shared__ uint32_t temp[];

	uint32_t blockID = blockIdx.x;
	uint32_t threadID = threadIdx.x;
	uint32_t blockOffset = blockID * n;

	uint32_t ai = threadID;
	uint32_t bi = threadID + (n / 2);
	uint32_t bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	uint32_t bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = input[blockOffset + ai + n * l];
	temp[bi + bankOffsetB] = input[blockOffset + bi + n * l];

	uint32_t offset = 1;
	for (uint32_t d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			uint32_t ai = offset * (2 * threadID + 1) - 1;
			uint32_t bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (threadID == 0) {
		sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	}

	for (uint32_t d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			uint32_t ai = offset * (2 * threadID + 1) - 1;
			uint32_t bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			uint32_t t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[blockOffset + ai] = temp[ai + bankOffsetA];
	output[blockOffset + bi] = temp[bi + bankOffsetB];
}

__global__ void add(uint32_t* output, uint32_t length, uint32_t* n) {
	uint32_t blockID = blockIdx.x;
	uint32_t threadID = threadIdx.x;
	uint32_t blockOffset = blockID * length;

	output[blockOffset + threadID] += n[blockID];
}

__global__ void add(uint32_t* output, uint32_t length, uint32_t* n1, uint32_t* n2) {
	uint32_t blockID = blockIdx.x;
	uint32_t threadID = threadIdx.x;
	uint32_t blockOffset = blockID * length;

	output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}




