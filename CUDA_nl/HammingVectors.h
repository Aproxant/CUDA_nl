#include <cstdint>
#include <iostream>
#pragma once
using namespace std;

class HammingVectors {  
public:

    int vector_count;
    int vector_len;
    int VLength; 
    uint32_t* data;
    uint32_t* invertedData;


    HammingVectors(int n, int l);
    /*
    void invertData();
    void radixSort(int flag);
    void xorData();
    void exclusiveRowOr(int flag);
    void exclusiveSum(int flag);
    void tobinaryArr(int flag);
    void exclusiveScanOr();
    int countPairs(int verbose);
    */
    ~HammingVectors();
private:
    //void countingSort(int digit, int bit,int flag);
    //int bitValue(int number, int bit);
    //uint64_t** radixSortNormal(uint64_t** arr,int* sortOr);
    //uint64_t** countingSortNormal(uint64_t** tmp, int digit, int bit,int *Or);
    //int countPairsSupply(uint64_t** arr, int size,int verbose,int* sortOr);
};




