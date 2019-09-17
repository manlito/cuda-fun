#include <iostream>
#include <vector>
#include "image.h"
#include "convolution_gpu.h"

int main(int argc, char **argv) {

    if (argc != 3) {
        std::cout << "Expected 2 filenames to be provided: 1st is input, 2nd is output" << std::endl;
        return 1;
    }
    std::string filename_input(argv[1]);
    std::cout << "Loading " << filename_input << std::endl;
    
    std::vector<unsigned char> image;
    int width;
    int height;
    int channels;
    
    read_jpeg_image(filename_input, image, width, height, channels);
    std::cout << "Read image: " << width << " x " << height << " x " << channels << std::endl;
    
    const int imageAllocSize = sizeof(float) * width * height * channels;
    float *image_device, *result_device;
    cudaMallocManaged(&image_device, imageAllocSize);
    cudaMallocManaged(&result_device, imageAllocSize);
    
    // Copy image to device
    for (int row = 0; row < height; row++) {
        float *row_ptr_device = &image_device[row * width * channels];
        unsigned char *row_ptr_host = &image[row * width * channels];
        for (int col = 0; col <  width * channels; col++, row_ptr_device++, row_ptr_host++)
            *row_ptr_device = static_cast<float>(*row_ptr_host);
    }
    cudaMemPrefetchAsync(&image_device, imageAllocSize, 0);
    cudaMemPrefetchAsync(&result_device, imageAllocSize, 0);
    
    blur_separable_gpu(image_device, width, height, channels, result_device);
    
    cudaDeviceSynchronize();
    std::vector<unsigned char> image_result;
    image_result.resize(width * height * channels);

    // Copy result to host
    for (int row = 0; row < height; row++) {
        float *row_ptr_device = &result_device[row * width * channels];
        unsigned char *row_ptr_host = &image_result[row * width * channels];
        for (int col = 0; col <  width * channels; col++, row_ptr_device++, row_ptr_host++)
            *row_ptr_host = static_cast<unsigned char>(std::min(255.f, std::max(0.f, *row_ptr_device)));
    }
    
    std::string filename_output(argv[2]);
    std::cout << "Saving " << filename_output << std::endl;
    write_jpeg_image(filename_output, image_result, width, height, channels);
    
    cudaFree(&image_device);
    cudaFree(&image_result);
    
    std::cout << "Done!" << std::endl;
    return 0;
}
