#include <iostream>
#include <vector>

__global__
void scale_kernel(float *const input_image,
                  const int size, 
                  float* result) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (;index < size; index += stride) {
         result[index] = 1.25f * input_image[index];
    }
}

void scale(float *const input_image,
           int width,
           int height,
           int channels,
           float* result) {
    int blockSize = 256;
    int numBlocks = (width * height * channels) / blockSize + 1;
    int size = width * height * channels;
    std::cout << "Launching kernel with " << numBlocks << " blocks @ " << blockSize << std::endl;
    scale_kernel<<<numBlocks, blockSize>>>(input_image, size, result);
    cudaDeviceSynchronize();
}
