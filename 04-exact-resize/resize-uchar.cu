#include <iostream>
#include <vector>
#include "error_check.h"

template<int LEVELS, int CHANNELS>
__global__
void
downscale_uchar(
    unsigned char *const input_image,
    const int source_width,
    const int source_height,
    const int target_width,
    const int target_height,
    unsigned char *result)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;
    const int source_step = source_width * CHANNELS;
    const int target_step = target_width * CHANNELS;
    const int block_step = blockDim.x * 2 * CHANNELS;

    extern __shared__ unsigned char stripe[];

    // Only populate pixels inside target photo
    if (x < target_width && y < target_height)
    {
        // Each thread needs to copy 2x2xCHANNELS pixels
        for (int row = 0; row < 2; row++)
        {
            for (int col = 0; col < 2; col++)
            {
                // Does an implicit border repeat
                const int safe_source_x = min(source_width, (x + col) * 2);
                const int safe_source_y = min(source_height, (y + row) * 2);
                for (int channel = 0; channel < CHANNELS; channel++)
                {
                    stripe[block_step * row + (threadIdx.x + col) * CHANNELS + channel] =
                        input_image[source_step * safe_source_y + safe_source_x * CHANNELS + channel];
                }
            }
        }
        __syncthreads();

        for (int channel = 0; channel < CHANNELS; channel++)
        {
            // Compute channel average. Unrolled double for
            float average =
                    (float) stripe[threadIdx.x * CHANNELS + channel] +
                    (float) stripe[(threadIdx.x + 1) * CHANNELS + channel] +
                    (float) stripe[block_step + threadIdx.x * CHANNELS + channel] +
                    (float) stripe[block_step + (threadIdx.x + 1) * CHANNELS + channel];
            result[y * target_step + x * CHANNELS + channel] =
                (unsigned char) min(255.0f, max(0.0f, (0.25f * average)));
            // result[y * target_step + x * CHANNELS + channel] = input_image[y * source_step + x * CHANNELS + channel];
        }

    }
}

template<int CHANNELS>
__global__
void
downscale_2_uchar(
    unsigned char *const input_image,
    const int source_width,
    const int source_height,
    const int target_width,
    const int target_height,
    unsigned char *result)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;
    const int source_step = source_width * CHANNELS;
    const int target_step = target_width * CHANNELS;

    // Only populate pixels inside target photo
    if (x < target_width && y < target_height)
    {
        // Build using kernel using coords
        //  [ y_i, x_i  y_i, x_j ]
        //  [ y_j, x_j  y_j, x_j ]
        int x_i = x * 2;
        int x_j = min(source_width, x_i + 1);
        int y_i = y * 2;
        int y_j = min(source_height, y_i + 1);
        for (int channel = 0; channel < CHANNELS; channel++)
        {
            // Compute channel average. Unrolled double for
            float average =
                    (float) input_image[source_step * y_i + x_i * CHANNELS + channel] +
                    (float) input_image[source_step * y_i + x_j * CHANNELS + channel] +
                    (float) input_image[source_step * y_j + x_i * CHANNELS + channel] +
                    (float) input_image[source_step * y_j + x_j * CHANNELS + channel];

            result[y * target_step + x * CHANNELS + channel] =
                (unsigned char) min(255.0f, max(0.0f, (0.25f * average)));
        }

    }
}
void
resize_uchar(unsigned char *const input_image,
             const int width,
             const int height,
             const int scale,
             unsigned char *result,
             cudaStream_t *stream)
{

    constexpr int BLOCK_SIZE = 64;
    dim3 block_size(BLOCK_SIZE, 1);
    const int target_width = width / scale;
    const int target_height = height / scale;
    const int shared_memory_size = BLOCK_SIZE * 2 * 2 * 3;
    dim3 grid_size((target_width + block_size.x - 1) / block_size.x, target_height);

    if (stream == nullptr)
    {
        if (scale == 2)
        {
            downscale_2_uchar<3> <<< grid_size, block_size >>> (
                input_image, width, height, target_width, target_height, result);
            gpuErrchk(cudaGetLastError());
        }
    }
    else
    {
        gpuErrchk(cudaGetLastError());
        if (scale == 2)
        {
            downscale_2_uchar<3> <<< grid_size, block_size, 0, *stream >>> (
                input_image, width, height, target_width, target_height, result);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaGetLastError());

        }
    }

}
