#include <iostream>
#include <vector>
#include <chrono>
#include <exception>
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

    std::vector<std::string> files = std::vector<std::string>();
    getdir(dirname_input, files);

    JobPool job_pool;
    job_pool.addJobs(files);
    std::mutex cout_mutex;

    typedef WorkerBaseUchar<3> Worker;

    auto gpu_work = [&job_pool, &dirname_input, &dirname_output, &cout_mutex](Worker *worker)
    {
        while (true)
        {
            auto nextJob = job_pool.getNextJob();

            if (nextJob == "")
            {
                {
                    std::lock_guard<std::mutex> lock_cout(cout_mutex);
                    std::cout << "Freeing images  @ " << worker->id << std::endl;
                }
                cudaFree(&(worker->input));
                cudaFree(&(worker->output));
                break;
            }

            if (isjpeg(nextJob))
            {

                {
                    std::lock_guard<std::mutex> lock_cout(cout_mutex);
                    std::cout << nextJob << " " << isjpeg(nextJob) << " @ " << worker->id << std::endl;
                }

                std::this_thread::sleep_for(std::chrono::seconds(1));
                const int scale = 2;

                int channels;
                read_jpeg_image_cu(dirname_input + "/" + nextJob,
                                   worker->source_width,
                                   worker->source_height,
                                   channels,
                                   worker->image_alloc_size_source,
                                   &worker->input,
                                   (void *)&worker->stream);

                if (channels != Worker::CHANNELS)
                {
                    throw std::runtime_error("Only images with 3 channels are supported");
                }
                {
                    std::lock_guard<std::mutex> lock_cout(cout_mutex);
                    std::cout << "Read image: " << worker->source_width << " x " << worker->source_height << " x "
                              << channels << std::endl;
                }

                worker->target_width = worker->source_width / scale;
                worker->target_height = worker->source_height / scale;
                {
                    std::lock_guard<std::mutex> lock_cout(cout_mutex);
                    std::cout << "Target size: " << worker->target_width << " x " << worker->target_height << std::endl;
                }

                int new_image_alloc_size_target = worker->target_width * worker->target_height * Worker::CHANNELS;
                if (new_image_alloc_size_target > worker->image_alloc_size_target)
                {
                    worker->image_alloc_size_target = new_image_alloc_size_target;
                    {
                        std::lock_guard<std::mutex> lock_cout(cout_mutex);
                        std::cout << "Allocating target " << worker->image_alloc_size_target << " @ " << worker->id
                                  << std::endl;
                    }
                    gpuErrchk(cudaMallocManaged(&(worker->output),
                                                worker->image_alloc_size_target,
                                                cudaMemAttachHost));
                    gpuErrchk(cudaStreamAttachMemAsync(worker->stream,
                                             worker->output,
                                             worker->image_alloc_size_target));
                }

                {
                    std::lock_guard<std::mutex> lock_cout(cout_mutex);
                    std::cout << "Calling resize @ " << worker->id << std::endl;
                }
                resize_uchar(worker->input,
                             worker->source_width,
                             worker->source_height,
                             scale,
                             worker->output,
                             &worker->stream);

                cudaStreamSynchronize(worker->stream);
                {
                    std::lock_guard<std::mutex> lock_cout(cout_mutex);
                    std::cout << "Saving " << nextJob << std::endl;
                }
                write_jpeg_image(dirname_output + "/" + nextJob,
                                 worker->output,
                                 worker->target_width,
                                 worker->target_height,
                                 Worker::CHANNELS);
            } // end isjpeg if

        }
    };

    std::vector<std::thread> workers;
    std::vector<Worker> workers_data;
    // Create workers
    for (int worker_index = 1; worker_index <= 16; worker_index++)
    {
        workers_data.push_back(Worker(worker_index));
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
    cudaDeviceReset();

    std::cout << "Done!" << std::endl;
    return 0;

}
