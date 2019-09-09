#include <iostream>

using namespace std;

__global__
void add_gpu(const int N, float *a, float *b, float *result) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Stride style loop
    const int stride = gridDim.x * blockDim.x;
    for (; index < N; index += stride) {
        result[index] = a[index] + b[index];
    }
    
    // Direct approach
    // if (index < N) {
    //     result[index] = a[index] + b[index];
    // }
    
}

int main() {
    
    int N = 1 << 20;
    
    cout << "Adding two " << N << " vectors" << std::endl;
    
    float *a, *b, *result;
    const int vectorMemSize = N * sizeof(float);
    
    // Forcing GPU allocation as we want to profile
    // time spent on kernel
    cudaMallocManaged(&a, vectorMemSize);
    cudaMallocManaged(&b, vectorMemSize);
    cudaMallocManaged(&result, vectorMemSize);
    
    for (int i = 0; i < N; i++) {
        a[i] = 1.0;
        b[i] = 3.0;
    }
    
    cudaMemPrefetchAsync(a, vectorMemSize, 0);
    cudaMemPrefetchAsync(b, vectorMemSize, 0);
    cudaMemPrefetchAsync(result, vectorMemSize, 0);

    const int blockSize = 128;
    const int numBlocks = N / blockSize + 1;
    // const int numBlocks = 1;

    add_gpu<<<numBlocks, blockSize>>>(N, a, b, result);
    
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
        cout << cudaGetErrorString(err);
    
    cudaDeviceSynchronize();
    
    double errorSum = 0;
    for (int i = 0; i < N; i++) {
        errorSum += abs(result[i] - 4.0);
    }
    cout << "Total error: " << errorSum << endl;
    
    cudaFree(&a);
    cudaFree(&b);
    cudaFree(&result);
    return 0;
}
