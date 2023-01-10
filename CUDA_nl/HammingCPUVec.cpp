#include <fstream>
#include "HammingCPUVec.h"
#include <iostream>

using namespace std;


HammingCPUVec::~HammingCPUVec()
    {
        delete[] data;
    }

bool HammingCPUVec::hammingDistanceOne(int i, int j) {
        long hamming = 0;
        // sum hamming distance for each sequence part
        for (long offset = 0; offset <VLenght; offset++) {
            // increment by hamming distance of the parts
            hamming += __popcnt(data[vector_count * offset + i] ^ data[vector_count * offset + j]);
            //if greater than one fail fast
            if (hamming > 1) return false;
        }
        return hamming == 1;
    }

uint64_t HammingCPUVec::hammingWithCPU(int verbose) {
        uint64_t counter = 0;
        if (verbose)
        {

            ofstream result("output.txt");
            for (long i = 0; i < vector_count; i++) {
                for (long j = i + 1; j < vector_count; j++) {
                    if (hammingDistanceOne(i, j))
                    {
                        result << i << " , " << j << endl;
                        counter++;
                    }

                }
            }
            result.close();
        }
        else
        {

            for (long i = 0; i < vector_count; i++) {
                for (long j = i + 1; j < vector_count; j++) {
                    if (hammingDistanceOne( i, j))
                    {
                        counter++;
                    }
                }
            }
        }

        return counter;
    }

void HammingCPUVec::parseNumber(string number, int bitsPerInt, int vecC, int vecNr)
{
    int bitPos = 0;
    int arrPos = 0;
    uint32_t pomValue = 0;
    for (int x = 0; x < number.length(); x++)
    {
        if (bitsPerInt > bitPos)
        {
            pomValue += (uint32_t)(number[x] - '0') << bitPos;
            bitPos++;
        }
        else
        {
            data[arrPos * vecC + vecNr] = pomValue;
            arrPos++;
            bitPos = 0;
            pomValue = 0;
            x--;
        }
    }
    if (bitPos != 0)
    {
        data[arrPos * vecC + vecNr] = pomValue;
    }
}

HammingCPUVec::HammingCPUVec(string path)
{


    ifstream myfile(path, ios::binary);

    if (!myfile.is_open())
    {
        cout << "Unable to open file " << path << endl;
        myfile.close();
        return;
    }



    int headerData[2];
    char no;

    myfile >> vector_count;
    myfile >> no;
    myfile >> vector_len;



    string line;


    VLenght = vector_len / WORD_BIT_LEN;
    if (vector_len % WORD_BIT_LEN)
        VLenght++;
    data = new uint32_t[vector_count * VLenght];


    string number;
    int arrPos = 0;
    int j = 0;


    while (myfile >> number)
    {
        parseNumber(number, WORD_BIT_LEN, vector_count, arrPos);

        arrPos++;
    }


    myfile.close();
}

