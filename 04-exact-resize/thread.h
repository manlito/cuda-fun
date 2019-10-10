#include <thread>
#include <mutex>
#include <deque>
#include <functional>

struct Worker {
    Worker(int id) 
        : id(id) {
        cudaStreamCreate(&stream);
    }
    
    void reserve() {
        image_device.resize(channels);
        result_device.resize(channels);
    }
    
    int id {0};

    std::vector<unsigned char> image;
    std::vector<float*> image_device;
    std::vector<float*> result_device;
    std::vector<unsigned char> image_result;
    
    int width;
    int height;
    int channels {3};
    int image_alloc_size_source {0};
    
    int target_width;
    int target_height;
    int image_alloc_size_target {0};
    
    std::thread thread;
    cudaStream_t stream;
};

class JobPool {
public:
    void addJobs(const std::vector<std::string> &files) {
        std::copy(files.begin(), files.end(), std::back_inserter(this->files));
    }
    std::string getNextJob() {
        std::lock_guard<std::mutex> lock(files_mutex);
        if (files.size()) {
            const auto next = files.front();
            files.pop_front();
            return next;
        } else {
            return "";
        }
    }
private:
    std::deque<std::string> files;
    std::mutex files_mutex;
};
