#include "grayscale.cuh"

#include <cstdio>
#include <stdexcept>

#include <cuda_helpers.h>
#include <cuda_runtime.h>

Image AllocHostImage(size_t width, size_t height, size_t channels) {
    Image image;
    image.width = width;
    image.height = height;
    image.channels = channels;
    image.pixels = new uint8_t[width * height * channels];
    image.stride = width * channels * sizeof(uint8_t);
    return image;
}

Image AllocDeviceImage(size_t width, size_t height, size_t channels) {
    Image image;
    image.width = width;
    image.height = height;
    image.channels = channels;
    uint8_t* a_device = nullptr;
    size_t pitch;
    CheckStatus(cudaMallocPitch(&a_device, &pitch, width * channels * sizeof(uint8_t), height));
    image.pixels = a_device;
    image.stride = pitch;
    return image;
}

void CopyImageHostToDevice(const Image& src_host, Image& dst_device) {
    // Image image_device = AllocDeviceImage(src_host.width, src_host.height, src_host.channels);
    CheckStatus(cudaMemcpy2D(dst_device.pixels, dst_device.stride, src_host.pixels, src_host.stride,
                             src_host.width * src_host.channels * sizeof(uint8_t), src_host.height,
                             cudaMemcpyHostToDevice));
    // dst_device = image_device;
}

void CopyImageDeviceToHost(const Image& src_device, Image& dst_host) {
    // Image image_host = AllocHostImage(src_device.width, src_device.height, src_device.channels);
    CheckStatus(cudaMemcpy2D(dst_host.pixels, dst_host.stride, src_device.pixels, src_device.stride,
                             src_device.width * src_device.channels * sizeof(uint8_t),
                             src_device.height, cudaMemcpyDeviceToHost));
    // dst_host = image_host;
}

__global__ void ConvertToGrayscale(uint8_t* rgb_pixels, uint8_t* grey_pixels, size_t width,
                                   size_t height, size_t grey_stride, size_t rgb_stride) {
    // size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    int64_t row_index = blockIdx.x / width;
    int64_t col_index = blockIdx.x % width;
    int64_t rgb_index = 3 * col_index + row_index * rgb_stride;

    grey_pixels[col_index + row_index * grey_stride] = 0.299 * rgb_pixels[rgb_index] +
                                                       0.587 * rgb_pixels[rgb_index + 1] +
                                                       0.114 * rgb_pixels[rgb_index + 2];
}

void ConvertToGrayscaleDevice(const Image& rgb_device_image, Image& gray_device_image) {
    if (rgb_device_image.height == 0 || rgb_device_image.width == 0) {
        return;
    }

    size_t nThreads = 1;
    size_t length = rgb_device_image.height * rgb_device_image.width;
    size_t nBlocks = (length + nThreads - 1) / nThreads;
    ConvertToGrayscale<<<nBlocks, nThreads>>>(rgb_device_image.pixels, gray_device_image.pixels,
                                              rgb_device_image.width, rgb_device_image.height,
                                              gray_device_image.stride, rgb_device_image.stride);
}

void FreeDeviceImage(const Image& image) {
    CheckStatus(cudaFree(image.pixels));
}

void FreeHostImage(const Image& image) {
    delete[] image.pixels;
    // free(image);
}
