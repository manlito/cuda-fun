#include <iostream>
#include <vector>
#include "error_check.h"

template <int KERNEL_SIZE>
__global__
void downscale(
    float *const input_image,
    const int width,
    const int height,
    const int target_width,
    const int target_height,
    float* result) {
    
    constexpr float NORMALIZATION_FACTOR = 1.0f/(KERNEL_SIZE*KERNEL_SIZE);
    const int y = blockIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Allocate as: (BLOCK_SIZE.x * KERNEL_SIZE) x (KERNEL_SIZE)
    extern __shared__ float input_shared[];
    
    // Populate shared memory with input image
    for (int kernel_y = 0; kernel_y < KERNEL_SIZE; kernel_y++) {
        const int row_source = min(height - 1, max(0, y * KERNEL_SIZE + kernel_y)) * width;
        const int row_shared = kernel_y * blockDim.x * KERNEL_SIZE;
        for (int kernel_x = 0; kernel_x < KERNEL_SIZE; kernel_x++) {
            input_shared[row_shared + threadIdx.x * KERNEL_SIZE + kernel_x] = 
                input_image[row_source + min(width - 1, max(0, x * KERNEL_SIZE + kernel_x))];
        }
    }
    __syncthreads();
    
    // Accumulate and store
    const int target_row = y * target_width;
    const int target_col = x;
    if (x < target_width) {
        float *result_ptr = &result[target_row + target_col];
        *result_ptr = 0;
        for (int kernel_y = 0; kernel_y < KERNEL_SIZE; kernel_y++) {
            const int row_shared = kernel_y * blockDim.x * KERNEL_SIZE;
            for (int kernel_x = 0; kernel_x < KERNEL_SIZE; kernel_x++) {
                *result_ptr += input_shared[row_shared + threadIdx.x * KERNEL_SIZE + kernel_x];
            }
        }
        *result_ptr *= NORMALIZATION_FACTOR;
    }
}

void resize(float *const input_image,
            const int width,
            const int height,
            const int scale,
            float* result, 
            cudaStream_t *stream) {
    
    constexpr int BLOCK_SIZE = 64;
    dim3 block_size(BLOCK_SIZE, 1);
    const int target_width = width / scale;
    const int target_height = height / scale;
    const int shared_memory_size = BLOCK_SIZE * scale * scale * sizeof(float);
    dim3 grid_size((target_width + block_size.x - 1) / block_size.x, target_height);

    std::cout << "Launching kernel " << grid_size.x << " x " << grid_size.y << std::endl;

    if (stream == nullptr) {
        if (scale == 2) {
            downscale<2><<<grid_size, block_size, shared_memory_size>>>(
                input_image, width, height, target_width, target_height, result);
        } else if (scale == 4) {
            downscale<4><<<grid_size, block_size, shared_memory_size>>>(
                input_image, width, height, target_width, target_height, result);
        } else if (scale == 8) {
            downscale<8><<<grid_size, block_size, shared_memory_size>>>(
                input_image, width, height, target_width, target_height, result);
        }
    } else {
        if (scale == 2) {
            downscale<2><<<grid_size, block_size, shared_memory_size, *stream>>>(
                input_image, width, height, target_width, target_height, result);
            gpuErrchk( cudaPeekAtLastError() );
        }
    }
    
}
