#pragma once
#include <iostream>
using namespace std;

class HammingCPUVec
{
public:
    int vector_count;
    int vector_len;
    int VLenght;
    const int WORD_BIT_LEN = 32;
    uint32_t* data;


    HammingCPUVec(string path);
    bool hammingDistanceOne(int i, int j);
    uint64_t HammingCPUVec::hammingWithCPU(int verbose);
    void parseNumber(string number, int bitsPerInt, int vecC, int vecNr);

    ~HammingCPUVec();

};
