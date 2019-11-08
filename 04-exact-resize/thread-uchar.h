#include <thread>
#include <mutex>
#include <deque>
#include <functional>

template <int DIMENSIONS=3>
struct GPUWorkerBaseUchar
{
    static constexpr int CHANNELS = DIMENSIONS;

    GPUWorkerBaseUchar()
    {
        cudaStreamCreate(&stream);
    }

    ~GPUWorkerBaseUchar()
    {
        cudaStreamDestroy(stream);
    }

    int id{0};

    unsigned char *input;
    unsigned char *output;

    int source_width;
    int source_height;
    int image_alloc_size_source{0};

    int target_width;
    int target_height;
    int image_alloc_size_target{0};

    cudaStream_t stream;
};


template <int DIMENSIONS=3>
struct CPUWorkerBaseUchar
{
    static constexpr int CHANNELS = DIMENSIONS;

    int id{0};

    std::vector<unsigned char> input;
    std::vector<unsigned char> output;

    int source_width;
    int source_height;

    int target_width;
    int target_height;
};
