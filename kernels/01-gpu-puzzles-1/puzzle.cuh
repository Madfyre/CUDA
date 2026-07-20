#pragma once
#include <cuda_helpers.h>

__global__ void Map(const float* data, float* out) {
    int idx = threadIdx.x;
    out[idx] = data[idx] + 10;
}

__global__ void Zip(const float* left, const float* right, float* out) {
    int idx = threadIdx.x;
    out[idx] = left[idx] + right[idx];
}

__global__ void Guard(const float* data, float* out, size_t size) {
    int idx = threadIdx.x;
    ////
    for (int i = idx; i < size; ++i) {
        out[i] = data[i] + 10;
    }
}

__global__ void Block(const float* data, float* out, float value, size_t size) {
    int idx = threadIdx.x;
    ////
    for (int i = idx; i < size; ++i) {
        out[i] = data[i] + value;
    }
}
