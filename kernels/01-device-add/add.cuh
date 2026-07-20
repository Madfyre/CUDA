#pragma once

#include <vector>

#include <cuda_helpers.h>
#include <cuda_runtime.h>

float* AllocDeviceVector(size_t num_elements) {
    float* a_device = nullptr;
    CheckStatus(cudaMalloc(&a_device, num_elements * sizeof(float)));
    return a_device;
}

void FreeDeviceVector(float* device_ptr) {
    CheckStatus(cudaFree(device_ptr));
}

void CopyHostVectorToDevice(const std::vector<float>& vector_host, float* dst_device_ptr) {
    CheckStatus(cudaMemcpy(dst_device_ptr, vector_host.data(), vector_host.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
}

std::vector<float> CopyDeviceVectorToHost(const float* ptr_device, size_t num_elements) {
    std::vector<float> b_host(num_elements);
    CheckStatus(cudaMemcpy(b_host.data(), ptr_device, num_elements * sizeof(float),
                           cudaMemcpyDeviceToHost));

    return b_host;
}

__global__ void AddDeviceElement(const float* left_device, const float* right_device,
                                 float* out_device, size_t num_elements) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_elements) {
        out_device[idx] = left_device[idx] + right_device[idx];
    }
}

void AddDeviceVectors(const float* left_device, const float* right_device, float* out_device,
                      size_t num_elements) {
    size_t nThreads = 1024;
    size_t nBlocks = (num_elements + nThreads - 1) / nThreads;
    AddDeviceElement<<<nBlocks, nThreads>>>(left_device, right_device, out_device, num_elements);
}
