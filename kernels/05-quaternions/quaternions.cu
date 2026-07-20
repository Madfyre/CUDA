#include "quaternions.cuh"

#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>

static constexpr size_t FACTOR = 8;

__device__ __forceinline__ Quaternion shfl_down_sync(unsigned mask, Quaternion q, unsigned delta) {
    return {
        .a = __shfl_down_sync(mask, q.a, delta),
        .b = __shfl_down_sync(mask, q.b, delta),
        .c = __shfl_down_sync(mask, q.c, delta),
        .d = __shfl_down_sync(mask, q.d, delta),
    };
}

__global__ void ReduceSumDeviceTreeWarpSync(size_t rows, size_t cols, const Quaternion* inp,
                                            size_t inp_stride, Quaternion* out) {
    extern __shared__ char smem[];
    Quaternion* quat_array = reinterpret_cast<Quaternion*>(smem);

    inp += blockIdx.x * inp_stride;

    Quaternion total_mul = {1, 0, 0, 0};
    QuaternionMultiplier mul;

    for (size_t idx = FACTOR * threadIdx.x; idx < min(FACTOR * (threadIdx.x + 1), cols); idx++) {
        total_mul = mul(total_mul, inp[idx]);
    }

#pragma unroll
    for (size_t shift = 1; shift < 32; shift *= 2) {
        total_mul = mul(total_mul, shfl_down_sync(0xffffffff, total_mul, shift));
    }

    if (threadIdx.x % 32 == 0) {
        quat_array[threadIdx.x / 32] = total_mul;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        Quaternion part_mul = {1, 0, 0, 0};
        if (threadIdx.x < blockDim.x / 32) {
            part_mul = quat_array[threadIdx.x];
        }

#pragma unroll
        for (size_t shift = 1; shift < 32; shift *= 2) {
            part_mul = mul(part_mul, shfl_down_sync(0xffffffff, part_mul, shift));
        }

        if (threadIdx.x == 0) {
            out[blockIdx.x] = part_mul;
        }
    }
}

void QuaternionsReduce(size_t rows, size_t cols, const Quaternion* inp, size_t inp_stride,
                       Quaternion* out, cudaStream_t stream) {

    dim3 block((cols + FACTOR - 1) / FACTOR);
    dim3 grid(rows);
    size_t shmem_size = ((block.x + 31) / 32) * sizeof(Quaternion);
    ReduceSumDeviceTreeWarpSync<<<grid, block, shmem_size, stream>>>(rows, cols, inp, inp_stride,
                                                                     out);
}
