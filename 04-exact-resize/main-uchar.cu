#include <iostream>
#include <vector>
#include "image.h"
#include "resize.h"
#include "error_check.h"

int main(int argc, char **argv) {

    if (argc != 3) {
        std::cout << "Expected 2 filenames to be provided: 1st is input, 2nd is output" << std::endl;
        return 1;
    }
    std::string filename_input(argv[1]);
    std::cout << "Loading " << filename_input << std::endl;
    
    unsigned char *input;
    int width;
    int height;
    int channels;
    const int scale = 2;
    
    int allocation_size_source = 0;
    read_jpeg_image_cu(filename_input, width, height, channels, allocation_size_source, &input);

    std::cout << "Read image: " << width << " x " << height << " x " << channels << std::endl;

    int target_width = width / scale;
    int target_height = height / scale;
    const int allocation_size_target = width * height * channels;
    std::cout << "Target size: " << target_width << " x " << target_height << std::endl;
    
    // Separate image channels
    unsigned char *output;
    gpuErrchk(cudaMallocManaged(&output, allocation_size_target));

    // Call resize kernel
    {
        resize_uchar(input, width, height, scale, output);
    }

    cudaDeviceSynchronize();

    // Dump output
    std::string filename_output(argv[2]);
    std::cout << "Saving " << filename_output << std::endl;
    write_jpeg_image(filename_output, output, target_width, target_height, channels);

    cudaFree(&input);
    cudaFree(&output);

    std::cout << "Done!" << std::endl;
    return 0;
}
