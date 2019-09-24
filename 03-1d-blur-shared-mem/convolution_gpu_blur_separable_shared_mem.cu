#include <iostream>
#include <vector>

#define BLOCK_SIZE 256
#define KERNEL_SIZE 9
#define HALF_KERNEL_SIZE 4

__global__
void convolution_1d_x_shared_memory(
    float *const input_image,
    const int width,
    const int height,
    float* result) {
    
    // Save input image in shared memory
    __shared__ float row_data[BLOCK_SIZE + 2 * HALF_KERNEL_SIZE];

    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (index > HALF_KERNEL_SIZE && index < (width - HALF_KERNEL_SIZE)) {
        
        const int row_offset = width * blockIdx.y;    
        const float normalizing_factor = 1.0f / static_cast<float>(KERNEL_SIZE);

        // Copy all needed by block only
        float *const row_input = &input_image[row_offset];

        row_data[threadIdx.x + HALF_KERNEL_SIZE] = row_input[index];
        
        // Special cases at block boundaries
        if (threadIdx.x == 0) {
            for (int kernel_index = 0; kernel_index < HALF_KERNEL_SIZE; kernel_index++) {
                row_data[kernel_index] = row_input[index - HALF_KERNEL_SIZE + kernel_index];
            } 
        } else if (threadIdx.x == blockDim.x - 1) {
            for (int kernel_index = 1; kernel_index <= HALF_KERNEL_SIZE; kernel_index++) {
                row_data[threadIdx.x + HALF_KERNEL_SIZE + kernel_index] = row_input[index + kernel_index];
            } 
        }
        __syncthreads();
        
        float *row_result = &result[row_offset];
        
        row_result[index] = 0;
        for (int kernel_offset = -HALF_KERNEL_SIZE; 
                kernel_offset <= HALF_KERNEL_SIZE; 
                kernel_offset++) {
            row_result[index] += row_data[threadIdx.x + HALF_KERNEL_SIZE + kernel_offset];
        }
        
        // normalize
        row_result[index] *= normalizing_factor;
    }
}

void blur_separable_gpu_shared_memory(
    float *const input_image,
    const int width,
    const int height,
    float* result) {
    
    dim3 block_size(BLOCK_SIZE, 1);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, height);
    
    std::cout << "Launching kernel " << grid_size.x << " x " << grid_size.y << std::endl;
    
    convolution_1d_x_shared_memory<<<grid_size, block_size>>>(
        input_image, width, height, result);
    
}
