#pragma once

#include <cstddef>

#include <cuda_helpers.h>
#include <cuda_runtime.h>

__global__ void Swap(char* str, size_t length) {
    size_t idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
    if (idx < length / 2) {
        char tmp = str[idx];
        str[idx] = str[length - idx - 1];
        str[length - idx - 1] = tmp;
    }
}

void ReverseDeviceStringInplace(char* str, size_t length) {
    if (length < 2) {
        return;
    }
    size_t nThreads = 1024;
    size_t nBlocks = ((length / 2) + nThreads - 1) / nThreads;
    Swap<<<nBlocks, nThreads>>>(str, length);
}
