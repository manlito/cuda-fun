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
    
    // Separate image channels
    const int image_alloc_size = sizeof(float) * width * height;
    std::vector<float*> image_device(channels);
    std::vector<float*> result_device(channels);
    for (int channel = 0; channel < channels; channel++) {
        cudaMallocManaged(&(image_device[channel]), image_alloc_size);
        cudaMallocManaged(&(result_device[channel]), image_alloc_size);
    }
    
    // Copy image to GPU
    for (int row = 0; row < height; row++) {
        unsigned char *row_ptr_host = &image[row * width * channels];
        std::vector<float*> row_ptr_device;
        for (int channel = 0; channel < channels; channel++) {
            row_ptr_device.push_back(&(image_device[channel])[row * width]);
        }
        
        for (int col = 0; col <  width; col++)
            for (int channel = 0; channel < channels; channel++) {
                *(row_ptr_device[channel]) = static_cast<float>(row_ptr_host[channel]);
                row_ptr_device[channel]++;
                row_ptr_host++;
            }
        
    }

    // Upload to GPU to prevent memory copy to be in profile
    for (int channel = 0; channel < channels; channel++) {
        cudaMemPrefetchAsync(&(image_device[channel]), image_alloc_size, 0);
        cudaMemPrefetchAsync(&(image_device[channel]), image_alloc_size, 0);
    }
    
    for (int channel = 0; channel < channels; channel++) {
        blur_separable_gpu_shared_memory(image_device[channel], width, height, result_device[channel]);
    }
    
    cudaDeviceSynchronize();
    std::vector<unsigned char> image_result;
    image_result.resize(width * height * channels);

    std::cout << "Reading result" << std::endl;
    // Copy image from GPU
    for (int row = 0; row < height; row++) {
        unsigned char *row_ptr_host = &image_result[row * width * channels];
        std::vector<float*> row_ptr_device;
        for (int channel = 0; channel < channels; channel++) {
            row_ptr_device.push_back(&(result_device[channel])[row * width]);
        }
        
        for (int col = 0; col <  width; col++)
            for (int channel = 0; channel < channels; channel++) {
                row_ptr_host[channel] = 
                    static_cast<unsigned char>(
                        std::max(0.f, 
                        std::min(255.f, *(row_ptr_device[channel]))));
                row_ptr_device[channel]++;
                row_ptr_host++;
            }
        
    }
    
    std::string filename_output(argv[2]);
    std::cout << "Saving " << filename_output << std::endl;
    write_jpeg_image(filename_output, image_result, width, height, channels);
    
    for (int channel = 0; channel < channels; channel++) {
        cudaFree(&(image_device[channel]));
        cudaFree(&(result_device[channel]));
    }
    
    std::cout << "Done!" << std::endl;
    return 0;
}
