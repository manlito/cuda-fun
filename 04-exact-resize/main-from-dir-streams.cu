#include <iostream>
#include <vector>
#include "image.h"
#include "resize.h"
#include "filesystem.h"
#include "thread.h"

int main(int argc, char **argv) {

    if (argc != 3) {
        std::cout << "Expected 2 directores to be provided: 1st is input, 2nd is output" << std::endl;
        return 1;
    }
    std::string dirname_input(argv[1]);
    std::string dirname_output(argv[2]);
    
    std::vector<std::string> files = std::vector<std::string>();
    getdir(dirname_input, files);
    
    JobPool job_pool;
    job_pool.addJobs(files);
    std::mutex cout_mutex;

    auto gpu_work = [&job_pool, &dirname_input, &dirname_output, &cout_mutex](Worker *worker_ptr) {
        auto &worker = *worker_ptr;
        while (true) {
            auto nextJob = job_pool.getNextJob();

            if (nextJob == "") {
                for (int channel = 0; channel < worker.channels && worker.image_alloc_size_source; channel++) {
                    cudaFree(&(worker.image_device[channel]));
                    cudaFree(&(worker.result_device[channel]));
                }
                break;
            }
            
            if (isjpeg(nextJob)) {

                if (!isjpeg(nextJob))
                    continue;

                {
                    std::lock_guard<std::mutex> lock_cout(cout_mutex);
                    std::cout << nextJob << " " << isjpeg(nextJob) << " @ " << worker.id << std::endl;
                }
                
                worker.reserve();
                
                const int scale = 2;
                
                read_jpeg_image(dirname_input + "/" + nextJob, worker.image, worker.width, worker.height, worker.channels);
                {
                    std::lock_guard<std::mutex> lock_cout(cout_mutex);
                    std::cout << "Read image: " << worker.width << " x " << worker.height << " x " << worker.channels << std::endl;
                }

                worker.target_width = worker.width / scale;
                worker.target_height = worker.height / scale;
                {
                    std::lock_guard<std::mutex> lock_cout(cout_mutex);
                    std::cout << "Target size: " << worker.target_width << " x " << worker.target_height << std::endl;
                }
                
                // Separate image channels
                int new_image_alloc_size_source = sizeof(float) * worker.width * worker.height;
                if (new_image_alloc_size_source > worker.image_alloc_size_source) {
                    worker.image_alloc_size_source = new_image_alloc_size_source;
                    for (int channel = 0; channel < worker.channels; channel++)
                        cudaMallocManaged(&(worker.image_device[channel]), worker.image_alloc_size_source);
                }
                int new_image_alloc_size_target = sizeof(float) * worker.target_width * worker.target_height;
                if (new_image_alloc_size_target > worker.image_alloc_size_target) {
                    worker.image_alloc_size_target = new_image_alloc_size_target;
                    for (int channel = 0; channel < worker.channels; channel++)
                        cudaMallocManaged(&(worker.result_device[channel]), worker.image_alloc_size_target);
                }
                
                // Copy image to GPU
                for (int row = 0; row < worker.height; row++) {
                    unsigned char *row_ptr_host = &worker.image[row * worker.width * worker.channels];
                    std::vector<float*> row_ptr_device;
                    for (int channel = 0; channel < worker.channels; channel++) {
                        row_ptr_device.push_back(&(worker.image_device[channel])[row * worker.width]);
                    }
                    
                    for (int col = 0; col <  worker.width; col++)
                        for (int channel = 0; channel < worker.channels; channel++) {
                            *(row_ptr_device[channel]) = static_cast<float>(*row_ptr_host);
                            row_ptr_device[channel]++;
                            row_ptr_host++;
                        }
                    
                }

                for (int channel = 0; channel < worker.channels; channel++) {
                    resize(worker.image_device[channel], worker.width, worker.height, scale, worker.result_device[channel], &worker.stream);
                }
                
                cudaStreamSynchronize(worker.stream);

                if (worker.image_result.size() != worker.target_width * worker.target_height * worker.channels)
                    worker.image_result.resize(worker.target_width * worker.target_height * worker.channels);

                {
                    std::lock_guard<std::mutex> lock_cout(cout_mutex);
                    std::cout << "Reading result" << std::endl;
                }

                // Copy image from GPU
                for (int row = 0; row < worker.target_height; row++) {
                    unsigned char *row_ptr_host = &worker.image_result[row * worker.target_width * worker.channels];
                    std::vector<float*> row_ptr_device;
                    for (int channel = 0; channel < worker.channels; channel++) {
                        row_ptr_device.push_back(&(worker.result_device[channel])[row * worker.target_width]);
                    }
                    
                    for (int col = 0; col < worker.target_width; col++)
                        for (int channel = 0; channel < worker.channels; channel++) {
                            *row_ptr_host = 
                                static_cast<unsigned char>(
                                    std::max(0.f, 
                                    std::min(255.f, *(row_ptr_device[channel]))));
                            row_ptr_device[channel]++;
                            row_ptr_host++;
                        }
                    
                }
                
                {
                    std::lock_guard<std::mutex> lock_cout(cout_mutex);
                    std::cout << "Saving " << nextJob << std::endl;
                }
                write_jpeg_image(dirname_output + "/" + nextJob, worker.image_result, worker.target_width, worker.target_height, worker.channels);             
            } // end isjpeg if
            
        }
    };
    
    std::vector<std::thread> workers;
    std::vector<Worker> workers_data;
    for (int worker_index = 0; worker_index < 6; worker_index++) {
        workers_data.emplace_back(Worker(worker_index));
        workers.emplace_back(std::thread(gpu_work, &workers_data[worker_index]));
    }
    
    for (auto &worker: workers) {
        worker.join();
    }
    cudaDeviceReset();
    
    std::cout << "Done!" << std::endl;
    return 0;
    
}
