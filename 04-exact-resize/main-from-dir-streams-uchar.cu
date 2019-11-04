#include <iostream>
#include <vector>
#include <chrono>
#include <exception>
#include <iomanip>
#include "image.h"
#include "resize.h"
#include "filesystem.h"
#include "thread.h"
#include "thread-uchar.h"
#include "error_check.h"

int
main(int argc, char **argv)
{

    if (argc != 3)
    {
        std::cout << "Expected 2 directores to be provided: 1st is input, 2nd is output" << std::endl;
        return 1;
    }
    std::string dirname_input(argv[1]);
    std::string dirname_output(argv[2]);
    const int scale = 4;
    const int gpu_workers_count = 8;

    std::vector<std::string> files = std::vector<std::string>();
    getdir(dirname_input, files);

    JobPool job_pool;
    job_pool.addJobs(files);
    std::mutex cout_mutex;

    typedef WorkerBaseUchar<3> Worker;

    auto gpu_work = [&job_pool, &dirname_input, &dirname_output, &cout_mutex, scale](Worker *worker)
    {
        auto safe_print = [&cout_mutex, &worker](const std::string &message) {
            std::lock_guard<std::mutex> lock_cout(cout_mutex);
            std::cout << "[" << std::setw (3)  << worker->id << "] " << message << std::endl;
        };
        while (true)
        {
            auto nextJob = job_pool.getNextJob();

            if (nextJob == "" && worker->image_alloc_size_source)
            {
                safe_print("Freeing memory");
                cudaFree(&(worker->input));
                cudaFree(&(worker->output));
                break;
            }

            if (isjpeg(nextJob))
            {

                safe_print("Next job is " + nextJob);

                auto input_image_allocation_function = [&worker](const int &allocation_size) -> void {
                    if (worker->image_alloc_size_source < allocation_size) {
                        // If already has some allocation clear
                        if (worker->image_alloc_size_source > 0) {
                            gpuErrchk(cudaFree(&worker->input));
                        }
                        gpuErrchk(cudaMallocManaged(&worker->input, allocation_size, cudaMemAttachHost));
                        gpuErrchk(cudaStreamAttachMemAsync(worker->stream, worker->input, allocation_size));
                        worker->image_alloc_size_source = allocation_size;
                    }
                };
                int channels;
                read_jpeg_image_cu(dirname_input + "/" + nextJob,
                                   worker->source_width,
                                   worker->source_height,
                                   channels,
                                   input_image_allocation_function,
                                   &worker->input);

                if (channels != Worker::CHANNELS)
                {
                    throw std::runtime_error("Only images with 3 channels are supported");
                }
                safe_print("Read image " + std::to_string(worker->source_width) + " x " + std::to_string(worker->source_height));

                worker->target_width = worker->source_width / scale;
                worker->target_height = worker->source_height / scale;

                int new_image_alloc_size_target = worker->target_width * worker->target_height * Worker::CHANNELS;
                if (new_image_alloc_size_target > worker->image_alloc_size_target)
                {
                    worker->image_alloc_size_target = new_image_alloc_size_target;
                    safe_print("Allocating target " + std::to_string(worker->image_alloc_size_target));
                    gpuErrchk(cudaMallocManaged(&(worker->output),
                                                worker->image_alloc_size_target,
                                                cudaMemAttachHost));
                    gpuErrchk(cudaStreamAttachMemAsync(worker->stream,
                                             worker->output,
                                             worker->image_alloc_size_target));
                }

                resize_uchar(worker->input,
                             worker->source_width,
                             worker->source_height,
                             scale,
                             worker->output,
                             &worker->stream);

                cudaStreamSynchronize(worker->stream);
                safe_print("Saving " + nextJob);
                write_jpeg_image(dirname_output + "/" + nextJob,
                                 worker->output,
                                 worker->target_width,
                                 worker->target_height,
                                 Worker::CHANNELS);
            } // end isjpeg if

        }
    };

    std::vector<std::thread> workers;
    std::vector<Worker> workers_data(gpu_workers_count);
    // Assign Ids
    for (int worker_index = 1; worker_index <= gpu_workers_count; worker_index++)
    {
        workers_data.at(worker_index - 1).id = worker_index;
    }
    // Launch them
    for (auto &worker_data : workers_data)
    {
        std::thread thread(gpu_work, &worker_data);
        workers.push_back(std::move(thread));
    }

    for (auto &worker: workers)
    {
        worker.join();
    }

    std::cout << "Done!" << std::endl;
    return 0;

}
