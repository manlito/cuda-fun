#include <iostream>
#include <vector>

__global__
void convolution_1d_y(float *const input_image,
                      int width, 
                      int channels,
                      int kernel_size,
                      float *result) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockIdx.y;
    const int stride = gridDim.x * blockDim.x;
    const int offset = row * width * channels;
    
    const float kernel_weight = 1.0f / kernel_size;
    const int half_kernel = kernel_size / 2;
   
    // Individual threads, each one dealing with 1 channel
    for (; col >= half_kernel * channels && col < (width - half_kernel) * channels; col += stride) {
        
        // Sum neighbors
        for (int i = -half_kernel * channels; i <= half_kernel * channels; i += channels)
            result[offset + col] += input_image[offset + col + i];
            
        // Reweigh
        result[offset + col] *= kernel_weight;
    }
}

void blur_separable_gpu(float *const input_image,
                        int width,
                        int height,
                        int channels,
                        float* result) {
    const dim3 blockSize(256, 1);
    const dim3 gridSize((width * 3 + blockSize.x - 1) / blockSize.x, height);
   
    convolution_1d_y<<<gridSize, blockSize>>>(input_image, width, 3, 9, result);
}
