#pragma once

#include <cstdio>
#include <stdexcept>

#include <cuda_helpers.h>
#include <cuda_device_runtime_api.h>

void CallHelloWorld() {
    printf("Hello, world!");
    cudaDeviceSynchronize();
}
