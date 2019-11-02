#include <iostream>
#include <vector>
#include <chrono>
#include <exception>
#include "image.h"
#include "resize.h"
#include "filesystem.h"
#include "thread.h"
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

    std::vector <std::string> files = std::vector<std::string>();
    getdir(dirname_input, files);

    JobPool job_pool;
    job_pool.addJobs(files);
    std::mutex cout_mutex;

    typedef WorkerBase<3> Worker;

    auto gpu_work = [&job_pool, &dirname_input, &dirname_output, &cout_mutex](Worker *worker)
    {
        while (true)
        {
            auto nextJob = job_pool.getNextJob();

            if (nextJob == "")
            {
                for (int channel = 0; channel < Worker::CHANNELS && worker->image_alloc_size_source; channel++)
                {
                    {
                        std::lock_guard <std::mutex> lock_cout(cout_mutex);
                        std::cout << "Freeing channel " << channel << " @ " << worker->id << std::endl;
                    }
                    cudaFree(&(worker->input_device[channel]));
                    cudaFree(&(worker->output_device[channel]));
                }
                break;
            }

            if (isjpeg(nextJob))
            {

                {
                    std::lock_guard <std::mutex> lock_cout(cout_mutex);
                    std::cout << nextJob << " " << isjpeg(nextJob) << " @ " << worker->id << std::endl;
                }

                std::this_thread::sleep_for(std::chrono::seconds(1));
                const int scale = 2;

                int channels;
                read_jpeg_image(dirname_input + "/" + nextJob,
                                worker->input_host,
                                worker->source_width,
                                worker->source_height,
                                channels);
                if (channels != Worker::CHANNELS)
                {
                    throw std::runtime_error("Only images with 3 channels are supported");
                }
                {
                    std::lock_guard <std::mutex> lock_cout(cout_mutex);
                    std::cout << "Read image: " << worker->source_width << " x " << worker->source_height << " x "
                              << channels << std::endl;
                }

                worker->target_width = worker->source_width / scale;
                worker->target_height = worker->source_height / scale;
                {
                    std::lock_guard <std::mutex> lock_cout(cout_mutex);
                    std::cout << "Target size: " << worker->target_width << " x " << worker->target_height << std::endl;
                }

                // Separate image channels
                int new_image_alloc_size_source = sizeof(float) * worker->source_width * worker->source_height;
                if (new_image_alloc_size_source > worker->image_alloc_size_source)
                {
                    worker->image_alloc_size_source = new_image_alloc_size_source;
                    {
                        std::lock_guard <std::mutex> lock_cout(cout_mutex);
                        std::cout << "Allocating source " << worker->image_alloc_size_source << " @ " << worker->id
                                  << std::endl;
                    }
                    for (int channel = 0; channel < Worker::CHANNELS; channel++)
                    {
                        gpuErrchk(cudaMallocManaged(&(worker->input_device[channel]),
                                                    worker->image_alloc_size_source,
                                                    cudaMemAttachHost));
                        cudaStreamAttachMemAsync(worker->stream,
                                                 worker->input_device[channel],
                                                 worker->image_alloc_size_source);
                    }
                }

                int new_image_alloc_size_target = sizeof(float) * worker->target_width * worker->target_height;
                if (new_image_alloc_size_target > worker->image_alloc_size_target)
                {
                    worker->image_alloc_size_target = new_image_alloc_size_target;
                    {
                        std::lock_guard <std::mutex> lock_cout(cout_mutex);
                        std::cout << "Allocating target " << worker->image_alloc_size_target << " @ " << worker->id
                                  << std::endl;
                    }
                    for (int channel = 0; channel < Worker::CHANNELS; channel++)
                    {
                        gpuErrchk(cudaMallocManaged(&(worker->output_device[channel]),
                                                    worker->image_alloc_size_target, cudaMemAttachHost));
                        cudaStreamAttachMemAsync(worker->stream,
                                                 worker->output_device[channel],
                                                 worker->image_alloc_size_target);
                    }
                }

                {
                    // Copy image to GPU
                    {
                        std::lock_guard <std::mutex> lock_cout(cout_mutex);
                        std::cout << "Copy image to GPU " << " @ " << worker->id << std::endl;
                    }
                    std::vector<float *> row_ptr_device;
                    for (int channel = 0; channel < Worker::CHANNELS; channel++)
                    {
                        row_ptr_device.push_back(worker->input_device[channel]);
                    }
                    for (int row = 0; row < worker->source_height; row++)
                    {
                        unsigned char
                            *row_ptr_host = &worker->input_host[row * worker->source_width * Worker::CHANNELS];

                        for (int col = 0; col < worker->source_width; col++)
                            for (int channel = 0; channel < Worker::CHANNELS; channel++)
                            {
                                row_ptr_device[channel][0] = static_cast<float>(*row_ptr_host);
                                row_ptr_device[channel]++;
                                row_ptr_host++;
                            }

                    }
                }

                for (int channel = 0; channel < Worker::CHANNELS; channel++)
                {
                    {
                        std::lock_guard <std::mutex> lock_cout(cout_mutex);
                        std::cout << "Calling resize ch " << channel << " @ " << worker->id << std::endl;
                    }
                    resize(worker->input_device[channel],
                           worker->source_width,
                           worker->source_height,
                           scale,
                           worker->output_device[channel],
                           &worker->stream);
                }

                cudaStreamSynchronize(worker->stream);

                if (worker->output_host.size() != worker->target_width * worker->target_height * Worker::CHANNELS)
                    worker->output_host.resize(worker->target_width * worker->target_height * Worker::CHANNELS);
                {
                    std::lock_guard <std::mutex> lock_cout(cout_mutex);
                    std::cout << "Reading result" << std::endl;
                }

                // Copy image from GPU
                for (int row = 0; row < worker->target_height; row++)
                {
                    unsigned char *row_ptr_host = &worker->output_host[row * worker->target_width * Worker::CHANNELS];
                    std::vector<float *> row_ptr_device;
                    for (int channel = 0; channel < Worker::CHANNELS; channel++)
                    {
                        row_ptr_device.push_back(&(worker->output_device[channel])[row * worker->target_width]);
                    }

                    for (int col = 0; col < worker->target_width; col++)
                        for (int channel = 0; channel < Worker::CHANNELS; channel++)
                        {
                            *row_ptr_host =
                                static_cast<unsigned char>(
                                    std::max(0.f,
                                             std::min(255.f, *(row_ptr_device[channel]))));
                            row_ptr_device[channel]++;
                            row_ptr_host++;
                        }

                }

                {
                    std::lock_guard <std::mutex> lock_cout(cout_mutex);
                    std::cout << "Saving " << nextJob << std::endl;
                }
                write_jpeg_image(dirname_output + "/" + nextJob,
                                 worker->output_host,
                                 worker->target_width,
                                 worker->target_height,
                                 Worker::CHANNELS);
            } // end isjpeg if

        }
    };

    std::vector <std::thread> workers;
    std::vector <Worker> workers_data;
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
