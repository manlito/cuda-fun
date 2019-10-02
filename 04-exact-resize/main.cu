#include <iostream>
#include <vector>
#include "image.h"
#include "resize.h"

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
    const int scale = 2;
    
    read_jpeg_image(filename_input, image, width, height, channels);
    std::cout << "Read image: " << width << " x " << height << " x " << channels << std::endl;

    int target_width = width / scale;
    int target_height = height / scale;
    std::cout << "Target size: " << target_width << " x " << target_height << std::endl;
    
    // Separate image channels
    const int image_alloc_size_source = sizeof(float) * width * height;
    const int image_alloc_size_target = sizeof(float) * target_width * target_height;
    std::vector<float*> image_device(channels);
    std::vector<float*> result_device(channels);
    for (int channel = 0; channel < channels; channel++) {
        cudaMallocManaged(&(image_device[channel]), image_alloc_size_source);
        cudaMallocManaged(&(result_device[channel]), image_alloc_size_target);
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
                *(row_ptr_device[channel]) = static_cast<float>(*row_ptr_host);
                row_ptr_device[channel]++;
                row_ptr_host++;
            }
        
    }

    // Upload to GPU to prevent memory copy to be accounted profile
    for (int channel = 0; channel < channels; channel++) {
        cudaMemPrefetchAsync(&(image_device[channel]), image_alloc_size_source, 0);
    }

    for (int channel = 0; channel < channels; channel++) {
        resize(image_device[channel], width, height, scale, result_device[channel]);
    }
    
    cudaDeviceSynchronize();
    std::vector<unsigned char> image_result;
    image_result.resize(target_width * target_height * channels);

    std::cout << "Reading result" << std::endl;
    // Copy image from GPU
    for (int row = 0; row < target_height; row++) {
        unsigned char *row_ptr_host = &image_result[row * target_width * channels];
        std::vector<float*> row_ptr_device;
        for (int channel = 0; channel < channels; channel++) {
            row_ptr_device.push_back(&(result_device[channel])[row * target_width]);
        }
        
        for (int col = 0; col < target_width; col++)
            for (int channel = 0; channel < channels; channel++) {
                *row_ptr_host = 
                    static_cast<unsigned char>(
                        std::max(0.f, 
                        std::min(255.f, *(row_ptr_device[channel]))));
                row_ptr_device[channel]++;
                row_ptr_host++;
            }
        
    }
    
    std::string filename_output(argv[2]);
    std::cout << "Saving " << filename_output << std::endl;
    write_jpeg_image(filename_output, image_result, target_width, target_height, channels);
    
    for (int channel = 0; channel < channels; channel++) {
        cudaFree(&(image_device[channel]));
        cudaFree(&(result_device[channel]));
    }
    
    std::cout << "Done!" << std::endl;
    return 0;
}
