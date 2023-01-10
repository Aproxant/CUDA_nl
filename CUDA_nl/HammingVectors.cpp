#pragma once
#include <fstream>
#include "HammingVectors.h"

HammingVectors::HammingVectors(int n, int l) {
    vector_count = n;
    vector_len = l;
    VLength = vector_len;


    data = new uint32_t[vector_count*vector_len];

    invertedData=new uint32_t[vector_count*vector_len];


}


HammingVectors::~HammingVectors()
{


    delete[] invertedData;
    delete[] data;

}