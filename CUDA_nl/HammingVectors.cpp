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

/*

void HammingVectors::invertData()
{
    for (int i = 0; i < vector_count; i++)
    {
        for (int j = 0; j < VLength; j++)
        {
            invertedData[i][j] = data[i][VLength - 1 - j];
            uint64_t rev = 0;

            for (int k = 0; k < 64; k++)
            {
                rev <<= 1;

                if ((invertedData[i][j] & 1) == 1)
                    rev ^= 1;

                invertedData[i][j] >>= 1;
            }

            invertedData[i][j] = rev;
        }
    }
}

int HammingVectors::bitValue(int number, int bit) {
    uint64_t mask = 1ULL << bit;
    if ((number & mask) != 0) {
        return 1;
    }
    return 0;
}

void HammingVectors::countingSort( int digit, int bit,int flag) {

    uint64_t** tmp=nullptr;
    int* tmpOrder = nullptr;
    if (flag)
    {
        tmp = data;
        tmpOrder = order;
    }
    else
    {
        tmp = invertedData;
        tmpOrder = invertedOrder;
    }
      
    uint64_t* counts = new uint64_t[]{ 0, 0 };

    for (int i = 0; i < vector_count; i++) {
        counts[bitValue(tmp[i][digit], bit)] += 1;
    }

    uint64_t* indices = new uint64_t[]{ 0, counts[0] };

    uint64_t** sortedArray = new uint64_t * [vector_count];

    int* sortOrder = new int[vector_count];
    for (int i = 0; i < vector_count; i++) {

        int itemBitValue = bitValue(tmp[i][digit], bit);

        sortOrder[indices[itemBitValue]] = tmpOrder[i];
        sortedArray[indices[itemBitValue]] = tmp[i];

        indices[itemBitValue] += 1;

    }

    tmp = sortedArray;
    tmpOrder = sortOrder;

    delete[] indices;
    delete[] counts;

    if (flag)
    {
        data=tmp;
        order=tmpOrder;
    }
    else
    {
        invertedData = tmp;
        invertedOrder = tmpOrder;
    }

}

void HammingVectors::radixSort(int flag)
{
    for (int i = VLength - 1; i >= 0; i--) {


        for (int j = 63; j >= 0; j--)
        {
            countingSort(i, j,flag);
        }
    }

}

void HammingVectors::xorData()
{
    for (int i = 0; i < vector_count - 1; i++)
    {
        for (int j = 0; j < VLength; j++)
        {
            data[i][j] = data[i][j] ^ data[i + 1][j];


            invertedData[i][j] = invertedData[i][j] ^ invertedData[i + 1][j];

        }
    }
    for (int i = 0; i < VLength; i++)
    {
        data[vector_count - 1][i] = 0;
        invertedData[vector_count - 1][i] = 0;
    }
}

void HammingVectors::tobinaryArr(int flag)
{
    int* tmp = nullptr;
    uint64_t** tmp_data = nullptr;
    if (flag)
    {
        tmp = arr;
        tmp_data = data;
    }
    else
    {
        tmp = invertedArr;
        tmp_data = invertedData;
    }

    for (int i = 0; i < vector_count; i++)
    {
        int invIter = 0;
        for (int j = 0; j < VLength; j++)
        {
            uint64_t mask = 1ULL;
            int counter = 0;
            int counterInv = 0;
            if (0 == j)
            {
                counterInv = 64*VLength-vector_len;
            }

            for (int k = 0; k < 64; k++)
            {
                
                if (flag)
                {
                    if (j * 64 + k == vector_len)
                        break;
                    if (tmp_data[i][j] & mask<<counter)
                        tmp[i * vector_len + (64 * j) + k] = 1;
                    else
                        tmp[i * vector_len + (64 * j) + k] = 0;
                    counter++;
                }
                else
                {                   
                    if (counterInv == 64 || invIter==vector_len)
                        break;
                    
                    if (tmp_data[i][j] & mask << counterInv)
                        tmp[i * vector_len + invIter] = 1;
                    else
                        tmp[i * vector_len + invIter] = 0;
                    counterInv++;
                    invIter++;

                }

            }
        }
    }

}

void HammingVectors::exclusiveScanOr()
{
    int* tmp = new int[vector_count * vector_len];
    int* tmpInv = new int[vector_count * vector_len];
    for (int i = 0; i < vector_count; i++)
    {
        tmp[i * vector_len] = 0;
        tmpInv[i * vector_len] = 0;
        tmp[i * vector_len+1] = arr[i*vector_len];
        tmpInv[i * vector_len+1] = invertedArr[i * vector_len];
    }
    for (int i = 0; i < vector_count ; i++)
    {
        for (int j = 2; j < vector_len; j++)
        {
            tmp[i * vector_len+j]= arr[i * vector_len+(j-2)] | arr[i * vector_len + (j - 1)];
            tmpInv[i * vector_len+j]= invertedArr[i * vector_len + (j - 2)] | invertedArr[i * vector_len + (j - 1)];
        }
    }
    arr = tmp;
    invertedArr = tmpInv;
}
void HammingVectors::exclusiveRowOr(int flag)
{
    int* tmp = nullptr;
    uint64_t** tmp_data = nullptr;
    if (flag)
    {
        tmp = arr;
        tmp_data = data;
    }
    else
    {
        tmp = invertedArr;
        tmp_data = invertedData;
    }
    for (int i = 0; i < vector_count; i++)
    {
        if (flag)
        {
            cout << endl;
            for (int j = 0; j < VLength; j++)
            {
                uint64_t mask = 1ULL;
                int counter = 0;
                for (int k = 0; k < 64; k++)
                {
                    if (j * 64 + k == vector_len)
                        break;
                    if (j == 0 && k == 0)
                    {
                        tmp[i * vector_len + (64 * j) + k] = 0;
                    }
                    else if (j == 0 && k == 1)
                    {
                        mask = 1;
                        if (tmp_data[i][j] & mask)
                            tmp[i * vector_len + (64 * j) + k] = 1;
                        else
                            tmp[i * vector_len + (64 * j) + k] = 0;
                    }
                    else if (k == 0)
                    {
                        mask = 1ULL;
                        if (tmp_data[i][j - 1] & mask << 63 || tmp_data[i][j - 1] & mask << 62)
                        {
                            tmp[i * vector_len + (64 * j) + k] = 1;
                        }
                        else
                        {
                            tmp[i * vector_len + (64 * j) + k] = 0;
                        }
                    }
                    else if (k == 1)
                    {
                        mask = 1ULL;
                        if (tmp_data[i][j - 1] & mask << 63 || tmp_data[i][j] & mask)
                        {
                            tmp[i * vector_len + (64 * j) + k] = 1;
                        }
                        else
                        {
                            tmp[i * vector_len + (64 * j) + k] = 0;
                        }
                    }
                    else
                    {
                        if (tmp_data[i][j] & (mask << counter) || tmp_data[i][j] & (mask << (counter + 1)))
                        {
                            tmp[i * vector_len + (64 * j) + k] = 1;
                        }
                        else
                        {
                            tmp[i * vector_len + (64 * j) + k] = 0;
                        }
                        counter++;
                    }
                    cout << tmp[i * vector_len + (64 * j) + k];
                }
            }
        }
        else
        {
            for (int j = 0; j < VLength; j++)
            {
                uint64_t mask = 1ULL;
                int counter = 0;
                for (int k = 0; k < 64; k++)
                {
                    if (j * 64 + k == vector_len)
                        break;
                    if (j == 0 && k == 0)
                    {
                        tmp[i * vector_len + (64 * j) + k] = 0;
                    }
                    else if (j == 0 && k == 1)
                    {
                        mask = 1ULL<63;
                        if (tmp_data[i][j] & mask)
                            tmp[i * vector_len + (64 * j) + k] = 1;
                        else
                            tmp[i * vector_len + (64 * j) + k] = 0;
                    }
                    else if (k == 0)
                    {
                        mask = 1ULL;
                        if (tmp_data[i][j-1] & mask << 63 || tmp_data[i][j-1] & mask << 62)
                        {
                            tmp[i * vector_len + (64 * j) + k] = 1;
                        }
                        else
                        {
                            tmp[i * vector_len + (64 * j) + k] = 0;
                        }
                    }
                    else if (k == 1)
                    {
                        mask = 1ULL;
                        if (tmp_data[i][VLength * 64 - 1 - j] & mask << 63 || tmp_data[i][VLength * 64 - 1 - j] & mask)
                        {
                            tmp[i * vector_len + (64 * j) + k] = 1;
                        }
                        else
                        {
                            tmp[i * vector_len + (64 * j) + k] = 0;
                        }
                    }
                    else
                    {
                        if (tmp_data[i][j] & (mask << counter) || tmp_data[i][j] & (mask << (counter + 1)))
                        {
                            tmp[i * vector_len + (64 * j) + k] = 1;
                        }
                        else
                        {
                            tmp[i * vector_len + (64 * j) + k] = 0;
                        }
                        counter++;
                    }
                    cout << tmp[i * vector_len + (64 * j) + k];
                }
            }
        }
    }
}


void HammingVectors::exclusiveSum(int flag)
{
    int* tmpData;
    if (flag)
        tmpData = arr;
    else
        tmpData = invertedArr;

    int* tmp = new int[vector_count * vector_len];

    for (int j = 0; j < vector_len; j++)
    {
        tmp[vector_len + j] = tmpData[j];
        tmp[j] = 0;
    }

    for (int i = 2; i < vector_count; i++)
    {
        for (int j = 0; j < vector_len; j++)
        {
            tmp[i * vector_len + j] = tmp[(i - 1) * vector_len + j] + tmpData[(i - 1) * vector_len + j];
        }
    }
    if (flag)
        arr = tmp;
    else
        invertedArr = tmp;
}


uint64_t** HammingVectors::countingSortNormal(uint64_t**tmp, int digit, int bit,int *sort) {

    uint64_t* counts = new uint64_t[]{ 0, 0 };

    for (int i = 0; i < vector_count*vector_len; i++) {
        counts[bitValue(tmp[i][digit], bit)] += 1;
    }

    uint64_t* indices = new uint64_t[]{ 0, counts[0] };

    uint64_t** sortedArray = new uint64_t * [vector_count*vector_len];
    int* newSort = new int[vector_count * vector_len];
    for (int i = 0; i < vector_count*vector_len; i++) {

        int itemBitValue = bitValue(tmp[i][digit], bit);

        sortedArray[indices[itemBitValue]] = tmp[i];

        newSort[indices[itemBitValue]] = sort[i];

        indices[itemBitValue] += 1;

    }
    
    copy(newSort, newSort+vector_count*vector_len, sort);

    
    delete[] indices;
    delete[] counts;

    return sortedArray;

}




uint64_t** HammingVectors::radixSortNormal(uint64_t** arr,int* sortOr)
{

    for (int place = 2; place >= 0; place--)
    {
        for (int i = 0; i < 32; i++)
        {
            arr=countingSortNormal(arr, place, i,sortOr);
        }
    }
    return arr;
        
}

int HammingVectors::countPairsSupply(uint64_t** arr, int size,int verbose,int *sorted)
{
    int pairs = 0;
    ofstream outfile("output.txt");
    if (verbose)
    {
        for (int i = 0; i < size - 1; i++)
        {
            if (arr[i][0] == arr[i + 1][0] && arr[i][1] == arr[i + 1][1] && arr[i][2] == arr[i + 1][2])
            {
                pairs++;
                outfile << sorted[i] << " , "<<sorted[i+1]<<std::endl;
            }
        }
        outfile.close();
    }
    else
    {
        for (int i = 0; i < size - 1; i++)
        {
            if (arr[i][0] == arr[i + 1][0] && arr[i][1] == arr[i + 1][1] && arr[i][2] == arr[i + 1][2])
                pairs++;
        }
    }
    return pairs;
}

int HammingVectors::countPairs(int verbose)
{
    uint64_t** finalArr = new uint64_t * [vector_count * vector_len];
    for (int i = 0; i < vector_count * vector_len; i++)
    {
        finalArr[i] = new uint64_t[3];
    }

    for (int i = 0; i < vector_len; i++)
    {
        for (int j = 0; j <vector_count ; j++)
        {
            finalArr[i * vector_count + j][0] = i;
            finalArr[i * vector_count + order[j]][1] = arr[j*vector_len + i];

            finalArr[i * vector_count + invertedOrder[j]][2] = invertedArr[(j+1) *vector_len-1 - i];

        }

    }

    int* sortedOrder = new int[vector_count * vector_len];

    for (int i = 0; i < vector_len; i++)
    {
        for (int j = 0; j < vector_count; j++)
            sortedOrder[i * vector_count + j] = j;
    }
    
    finalArr=radixSortNormal(finalArr,sortedOrder);
    
    cout << "-------------final array sorted-------------" << endl;
    for (int j = 0; j < vector_count; j++)
        cout << order[j] << ",";

    cout << endl;
    for (int j = 0; j < vector_count; j++)
        cout << invertedOrder[j] << ",";

    cout << endl;

    for (int i = 0; i < vector_count * vector_len; i++)
    {
        cout <<sortedOrder[i]<<" , "<< finalArr[i][0] << " , " << finalArr[i][1] << " , " << finalArr[i][2] << endl;
    }
    
    return countPairsSupply(finalArr, vector_count * vector_len,verbose,sortedOrder);
}
*/
HammingVectors::~HammingVectors()
{


    delete[] invertedData;
    delete[] data;

}