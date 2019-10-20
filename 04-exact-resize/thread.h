#include <thread>
#include <mutex>
#include <deque>
#include <functional>

template <int DIMENSIONS=3>
struct WorkerBase
{
    WorkerBase(int id)
        : id(id)
    {
        cudaStreamCreate(&stream);
    }

    int id{0};

    std::vector<unsigned char> input_host;
    float * input_device[DIMENSIONS];
    std::vector<unsigned char> output_host;
    float * output_device[DIMENSIONS];

    int source_width;
    int source_height;
    static constexpr int CHANNELS = DIMENSIONS;
    int image_alloc_size_source{0};

    int target_width;
    int target_height;
    int image_alloc_size_target{0};

    cudaStream_t stream;
};

class JobPool
{
public:
    void
    addJobs(const std::vector <std::string> &files)
    {
        std::copy(files.begin(), files.end(), std::back_inserter(this->files));
    }
    std::string
    getNextJob()
    {
        std::lock_guard <std::mutex> lock(files_mutex);
        if (files.size())
        {
            const auto next = files.front();
            files.pop_front();
            return next;
        }
        else
        {
            return "";
        }
    }
private:
    std::deque <std::string> files;
    std::mutex files_mutex;
};
