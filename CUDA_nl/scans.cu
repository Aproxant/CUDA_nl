//Source https://github.com/mattdean1/cuda
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdint.h"
#include "fastSum.cuh"
#include "scans.cuh"

#define checkCudaError(o, l) _checkCudaError(o, l, __func__)

int THREADS_PER_BLOCK = 512;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;



uint32_t nextPowerOfTwo(uint32_t x) {
	uint32_t power = 1;
	while (power < x) {
		power *= 2;
	}
	return power;
}

void scanSmallDeviceArray(uint32_t* d_out, uint32_t* d_in, uint32_t length, uint32_t l) {
	uint32_t powerOfTwo = nextPowerOfTwo(length);

	prescan_arbitrary << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(uint32_t) >> > (d_out, d_in, length, powerOfTwo, l);



}

void scanLargeEvenDeviceArray(uint32_t* d_out, uint32_t* d_in, uint32_t length, uint32_t l) {
	const uint32_t blocks = length / ELEMENTS_PER_BLOCK;
	const uint32_t sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(uint32_t);

	uint32_t* d_sums, * d_incr;

	cudaMalloc((void**)&d_sums, blocks * sizeof(uint32_t));
	cudaMalloc((void**)&d_incr, blocks * sizeof(uint32_t));




	prescan_large << <blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize >> > (d_out, d_in, ELEMENTS_PER_BLOCK, d_sums, l);



	const uint32_t sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
		// perform a large scan on the sums arr
		scanLargeDeviceArray(d_incr, d_sums, blocks, l);
	}
	else {
		// only need one block to scan sums arr so can use small scan
		scanSmallDeviceArray(d_incr, d_sums, blocks, l);
	}

	add << <blocks, ELEMENTS_PER_BLOCK >> > (d_out, ELEMENTS_PER_BLOCK, d_incr);


	cudaFree(d_sums);
	cudaFree(d_incr);
}

void scanLargeDeviceArray(uint32_t* d_out, uint32_t* d_in, uint32_t length, uint32_t l) {
	uint32_t remainder = length % (ELEMENTS_PER_BLOCK);


	if (remainder == 0) {
		scanLargeEvenDeviceArray(d_out, d_in, length, l);
	}
	else {
		// perform a large scan on a compatible multiple of elements
		uint32_t lengthMultiple = length - remainder;
		scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, l);

		// scan the remaining elements and add the (inclusive) last element of the large scan to this
		uint32_t* startOfOutputArray = &(d_out[lengthMultiple]);
		scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder, l);

		add << <1, remainder >> > (startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
	}
}