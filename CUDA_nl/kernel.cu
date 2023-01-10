
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <chrono>
#include<string>
#include <bitset>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <chrono>

#pragma once
#include "HammingVectors.h"
#include "scans.cuh"
#include "HammingCPUVec.h"

#define ThreadNr 1024


using namespace std;
using namespace std::chrono;


cudaError_t findPairs(HammingVectors *vec);

void PrintVector(uint32_t* vec,int n,int l) {
    for (int i = 0; i < n; i++) {
        cout << endl;
        for (int j = 0; j < l; j++)
            cout << vec[i + (j* n)];
    }
    /*
    cout << endl;
    cout << endl;
    for (int i = 0; i < vec->vector_count; i++) {
        cout << endl;
        for (int j = 0; j < vec->vector_len; j++)
            cout << vec->invertedData[i + (j * vec->vector_count)];
    }*/
}

bool LoadSequences(string path, HammingVectors*& vec)
{
    ifstream myfile(path, ifstream::binary);

    if (!myfile.is_open())
    {
        cout << "Unable to open file " << path << endl;
        myfile.close();
        return false;
    }

    int headerData[2];
    char no;

    myfile >> headerData[0];
    myfile >> no;
    myfile >> headerData[1];

    if (nullptr != vec)
        delete vec;

    vec = new HammingVectors(headerData[0], headerData[1]);
    char c;
    for (int i = 0; i < vec->vector_count; i++)
    {
        for (int j = 0; j < vec->vector_len; j++)
        {
            myfile >> c;
            vec->data[i + (j * vec->vector_count)] = c - '0';
            vec->invertedData[(vec->vector_len-1-j) * vec->vector_count + i] = c - '0';
        }

    }


    myfile.close();
    return true;
}




int main(int argc, char** argv)
{
    int verbose = 0;
    int cpu = 0;
    /*
    if (argc < 2)
    {
        printf("Provide file with data\n");
        return 1;
    }

    if (argc == 4)
    {
        cpu = 1;
        verbose = 1;
    }
    if (argc == 3)
    {
        cpu = 1;
    }
    */


    //Fast IO initialization
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
    //Reading input data

    HammingVectors* hamSet = nullptr;

    cout << "Wczytywanie danych" << endl;
    auto start = high_resolution_clock::now();

    
    if (!LoadSequences("test1.dat", hamSet))
        return;

    auto stop = high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = stop - start;
    cout << "Wczytywanie ukonczone. Time: " << elapsed_seconds.count() << " [s]" << endl;

    findPairs(hamSet);
    cout << endl;

    if (cpu)
    {
        HammingCPUVec cpuHam=HammingCPUVec("test1.dat");
        cout << "CPU Hamming" << endl;
        auto start = high_resolution_clock::now();
        cout << "Pairs found: " << cpuHam.hammingWithCPU(verbose) << endl;
        auto stop = high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = stop - start;

        cout << "Time: " << elapsed_seconds.count() << "[s]" << endl;
    }


    return 0;
}

void raddixSort(uint32_t *data,uint32_t* perm,uint32_t* dev_row,int n,int l)
{
    thrust::sequence(thrust::device,perm, perm+ n, 0u);

    for (int i = l - 1; i >= 0; i--)
    {
        thrust::gather(thrust::device,perm, perm + n, data + i * n, dev_row);
       
        thrust::stable_sort_by_key(thrust::device,dev_row, dev_row+ n, perm);       
    }

    for (int i = 0; i < l; i++)
    {
        thrust::gather(thrust::device, perm, perm + n, data + i * n, dev_row);
        
        thrust::copy(thrust::device, dev_row, dev_row + n, data + i * n);
    }
}




__global__ void xorKenrel(uint32_t* data, uint32_t* invdata, int n, int l)
{
    int idX = threadIdx.x + blockIdx.x * blockDim.x;
    if (idX < n)
    {
        for (int i = 0; i < l; i++)
        {
            
            if (idX == n - 1)
            {
                data[idX + n * i] = 0;

                invdata[idX + n * i] = 0;
            }
            else
            {


                data[idX + n * i] = data[idX + n * i] ^ data[idX + 1 + n * i];


                invdata[idX + n * i] = invdata[idX + n * i] ^ invdata[idX + 1 + n * i];

            }
            __syncthreads();
        }
    }
}

__global__ void xor2Kenrel(uint32_t* data, uint32_t* tmp, int n, int l)
{
    int idX = threadIdx.x + blockIdx.x * blockDim.x;
    if (idX < n-1)
    {
        for (int i = 0; i < l; i++)
        {
                tmp[idX + n * i] = data[idX + n * i] ^ data[idX + 1 + n * i];
        }
    }
}

void xorVectors(uint32_t* data, uint32_t* dataInv, int n, int l)
{
    uint32_t* tmp = nullptr;
    cudaMalloc((void**)&tmp, n * l * sizeof(uint32_t));

    
    int blockNr = n / ThreadNr;
    if (n % ThreadNr != 0)
        blockNr++;

    thrust::fill(thrust::device, tmp, tmp + n * l, 0u);
    xor2Kenrel << <blockNr, ThreadNr >> > (data, tmp, n, l);
    thrust::copy(thrust::device, tmp, tmp + n*l, data);

    thrust::fill(thrust::device, tmp, tmp + n * l, 0u);
    xor2Kenrel << <blockNr, ThreadNr >> > (dataInv, tmp, n, l);
    thrust::copy(thrust::device, tmp, tmp + n * l, dataInv);

    cudaFree(tmp);


}

__global__ void  exclusiveOrKernel(uint32_t* data, int n, int l)
{
    uint32_t tmp;
    uint32_t tmp2;
    int idX = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idX < n)
    {

        tmp = data[idX + n];
        tmp2= data[idX];
        data[idX + n] = data[idX];
        data[idX] = 0;


        for (int i = 2; i < l; i++)
        {
            if (tmp2 || tmp)
            {
                tmp2 = 1u;
                tmp = 1u;

                data[idX + n * i] = 1u;
            }
            else
            {
                tmp2 = tmp;
                tmp = data[idX + n * i];
                data[idX + n * i] = 0u;
            }
            
        }

    }

}



void setRightOrder(uint32_t* data, uint32_t* dev_perm,uint32_t* dev_row, int n, int l)
{
    thrust::sequence(thrust::device, dev_row, dev_row + n, 0u);
    
    thrust::stable_sort_by_key(thrust::device, dev_perm, dev_perm + n, dev_row);

    for (int i = 0; i < l; i++)
    {
        thrust::gather(thrust::device, dev_row, dev_row + n, data + i * n, dev_perm);

        thrust::copy(thrust::device, dev_perm, dev_perm + n, data + i * n);
    }

    thrust::copy(thrust::device, dev_row, dev_row + n, dev_perm);

}

void create_table(uint32_t* data,uint32_t* data_inv,uint32_t* dev_final,uint32_t* dev_row, int n, int l)
{
    thrust::copy(thrust::device, data , data + n*l, dev_final+l*n);

    for (int i = 0; i < l; i++)
    {
        thrust::copy(thrust::device, data_inv + (l-i-1) * n, data_inv + (l - i) * n, (dev_final + n * l * 2) + (i * n));
        thrust::fill(thrust::device, dev_final + i * n, dev_final + (i+1) * n, i);
    }        
}


__global__ void countPairsKernel(uint32_t* data,uint64_t* pairs_count,int n, int l)
{
    int idX = blockIdx.x * blockDim.x + threadIdx.x;
    if (idX < n * l - 1)
        if (data[idX] == data[idX + 1] && data[idX + n * l] == data[idX + 1 + n * l] && data[idX + n * l * 2] == data[idX + 1 + n * l * 2])
            atomicAdd(pairs_count, 1);
}

cudaError_t findPairs(HammingVectors* vec)
{
    uint32_t* dev_vec = nullptr;
    uint32_t* dev_invVec = nullptr;


    uint32_t* dev_row = nullptr;
    uint32_t* dev_perm = nullptr;
    uint32_t* dev_invPerm = nullptr;

    uint32_t* dev_finalTable = nullptr;

    uint64_t* dev_pair_count = nullptr;


    cudaError_t cudaStatus;
    cudaEvent_t start, stop;

    float time = 0, time_temp;
    uint64_t pairs;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_vec, vec->vector_len*vec->vector_count * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_invVec, vec->vector_len * vec->vector_count * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_row, vec->vector_count * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_perm,  vec->vector_count * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_invPerm, vec->vector_count * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_finalTable, vec->vector_count*vec->vector_len*3 * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_pair_count,sizeof(uint64_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemset(dev_pair_count,0, sizeof(uint64_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMamset failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_vec, vec->data, vec->vector_len * vec->vector_count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_invVec, vec->invertedData, vec->vector_len * vec->vector_count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    cout << "Cuda start" << endl;

    int blockNr = vec->vector_count / ThreadNr;
    if (vec->vector_count % ThreadNr != 0)
        blockNr++;

    cudaEventRecord(start, 0);

    raddixSort(dev_vec, dev_perm,dev_row, vec->vector_count, vec->vector_len);
    raddixSort(dev_invVec, dev_invPerm,dev_row, vec->vector_count, vec->vector_len);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_temp, start, stop);
    time += time_temp;
    printf("Sorting time %f [s]\n", time_temp / 1000);

    cudaEventRecord(start, 0);

    xorVectors(dev_vec, dev_invVec, vec->vector_count, vec->vector_len);
    //xorKenrel<<<blockNr,ThreadNr>>>(dev_vec,dev_invVec, vec->vector_count, vec->vector_len);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_temp, start, stop);
    time += time_temp;
    printf("XOR time %f [s]\n", time_temp / 1000);

    cudaEventRecord(start, 0);

    exclusiveOrKernel <<<blockNr, ThreadNr >> > (dev_vec, vec->vector_count, vec->vector_len);
    exclusiveOrKernel << <blockNr, ThreadNr >> > (dev_invVec, vec->vector_count, vec->vector_len);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_temp, start, stop);
    time += time_temp;
    printf("Exclusive OR time %f [s]\n", time_temp / 1000);

    cudaEventRecord(start, 0);


    
    if (vec->vector_count > ThreadNr)
    {
        for (int i = 0; i < vec->vector_len; i++)
        {
            thrust::fill(thrust::device, dev_row, dev_row + vec->vector_count,0u);
            scanLargeDeviceArray(dev_row, dev_vec, vec->vector_count, i);
            thrust::copy(thrust::device, dev_row, dev_row + vec->vector_count, dev_vec + i * vec->vector_count);

            thrust::fill(thrust::device, dev_row, dev_row + vec->vector_count , 0u);
            scanLargeDeviceArray(dev_row, dev_invVec, vec->vector_count, i);
            thrust::copy(thrust::device, dev_row, dev_row + vec->vector_count, dev_invVec + i * vec->vector_count);
        }
        
    }
    else
    {
        for (int i = 0; i < vec->vector_len; i++)
        {
            scanSmallDeviceArray(dev_row, dev_vec, vec->vector_count, i);
            thrust::copy(thrust::device, dev_row, dev_row + vec->vector_count, dev_vec + i * vec->vector_count);

            scanSmallDeviceArray(dev_row, dev_invVec, vec->vector_count, i);
            thrust::copy(thrust::device, dev_row, dev_row + vec->vector_count, dev_invVec + i * vec->vector_count);
        }
        
    }

        
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_temp, start, stop);
    time += time_temp;
    printf("Exclusive SUM time %f [s]\n", time_temp / 1000);

    cudaEventRecord(start, 0);

    setRightOrder(dev_vec, dev_perm, dev_row, vec->vector_count, vec->vector_len);
    setRightOrder(dev_invVec, dev_invPerm, dev_row, vec->vector_count, vec->vector_len);



    //Final table

    create_table(dev_vec, dev_invVec, dev_finalTable, dev_row, vec->vector_count, vec->vector_len);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_temp, start, stop);
    time += time_temp;
    printf("Generating final table %f [s]\n", time_temp / 1000);

    cudaFree(dev_vec);
    cudaFree(dev_invVec);
    cudaFree(dev_perm);
    cudaFree(dev_invPerm);
    cudaFree(dev_row);


    cudaStatus = cudaMalloc((void**)&dev_row, vec->vector_len * vec->vector_count * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_perm, vec->vector_len * vec->vector_count * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    blockNr = vec->vector_count * vec->vector_len / ThreadNr;
    if (vec->vector_count * vec->vector_len % ThreadNr != 0)
        blockNr++;


    cudaEventRecord(start, 0);

    raddixSort(dev_finalTable, dev_perm, dev_row, vec->vector_count*vec->vector_len,3);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_temp, start, stop);
    time += time_temp;
    printf("Sorting final table %f [s]\n", time_temp / 1000);

   

    cudaEventRecord(start, 0);

    countPairsKernel<<<blockNr,ThreadNr>>>(dev_finalTable, dev_pair_count, vec->vector_count , vec->vector_len);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_temp, start, stop);
    time += time_temp;
    printf("Finding pairs %f [s]\n", time_temp / 1000);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "findPairsKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching findPairsKernel!\n", cudaStatus);
        goto Error;
    }


    cudaStatus = cudaMemcpy(&pairs, dev_pair_count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
  

    printf("Cuda done. Time %f [s]\n", time / 1000);

    cout << "Result. Pairs found: " << pairs << endl;

        

Error:
    cudaFree(dev_vec);
    cudaFree(dev_invVec);
    cudaFree(dev_row);
    cudaFree(dev_perm);
    cudaFree(dev_invPerm);
    cudaFree(dev_finalTable);
    cudaFree(dev_pair_count);


    return cudaStatus;
}
