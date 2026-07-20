#include <cuda_fp16.h>

#include <cuda_helpers.h>

constexpr int TILE = 32;

__global__ void TransposeKernel(const __half* input, size_t input_stride, __half* output,
                                size_t output_stride, size_t num_rows, size_t num_cols) {

    __shared__ __half tile[TILE][TILE + 1];

    size_t x = blockIdx.x * TILE + threadIdx.x;
    size_t y = blockIdx.y * TILE + threadIdx.y;

    if (x < num_cols && y < num_rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * input_stride + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    if (x < num_rows && y < num_cols) {
        output[y * output_stride + x] = tile[threadIdx.x][threadIdx.y];
    }
}

void TransposeDevice(const __half* input_device, size_t input_stride, __half* output_device,
                     size_t output_stride, size_t num_rows, size_t num_cols) {
    dim3 block(TILE, TILE);
    dim3 grid((num_cols + TILE - 1) / TILE, (num_rows + TILE - 1) / TILE);

    TransposeKernel<<<grid, block>>>(input_device, input_stride, output_device, output_stride,
                                     num_rows, num_cols);

    // CheckStatus(cudaDeviceSynchronize());
}
