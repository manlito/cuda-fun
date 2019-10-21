#include <thread>
#include <mutex>
#include <deque>
#include <functional>

template <int DIMENSIONS=3>
struct WorkerBaseUchar
{
    WorkerBaseUchar(int id)
        : id(id)
    {
        cudaStreamCreate(&stream);
    }

    int id{0};

    unsigned char *input;
    unsigned char *output;

    int source_width;
    int source_height;
    static constexpr int CHANNELS = DIMENSIONS;
    int image_alloc_size_source{0};

    int target_width;
    int target_height;
    int image_alloc_size_target{0};

    cudaStream_t stream;
};

