#include <iostream>
#include <vector>

template<int CHANNELS>
void
downscale_2_uchar(
    unsigned char *const input_image,
    const int source_width,
    const int source_height,
    const int target_width,
    const int target_height,
    unsigned char *result)
{
    const int source_step = source_width * CHANNELS;
    const int target_step = target_width * CHANNELS;

    for (int y = 0; y < target_height; y++)
        for (int x = 0; x < target_width; x++)
        {
            // Build using kernel using coords
            //  [ y_i, x_i  y_i, x_j ]
            //  [ y_j, x_j  y_j, x_j ]
            int x_i = x * 2;
            int x_j = std::min(source_width, x_i + 1);
            int y_i = y * 2;
            int y_j = std::min(source_height, y_i + 1);
            for (int channel = 0; channel < CHANNELS; channel++)
            {
                // Compute channel average. Unrolled double for
                float average =
                    (float) input_image[source_step * y_i + x_i * CHANNELS + channel] +
                        (float) input_image[source_step * y_i + x_j * CHANNELS + channel] +
                        (float) input_image[source_step * y_j + x_i * CHANNELS + channel] +
                        (float) input_image[source_step * y_j + x_j * CHANNELS + channel];

                result[y * target_step + x * CHANNELS + channel] =
                    (unsigned char) std::min(255.0f, std::max(0.0f, (0.25f * average)));
            }

        }
}


template<int CHANNELS, int SCALE>
void
downscale_level_uchar(
    unsigned char *const input_image,
    const int source_width,
    const int source_height,
    const int target_width,
    const int target_height,
    unsigned char *result)
{
    const int source_step = source_width * CHANNELS;
    const int target_step = target_width * CHANNELS;

    for (int y = 0; y < target_height; y++)
        for (int x = 0; x < target_width; x++)
        {
            // Copy pixels to average to this area
            float reduction_area[SCALE][SCALE * CHANNELS];

            for (int row = 0; row < SCALE; row++)
            {
                int safe_y = std::min(y * SCALE + row, source_height - 1);
                for (int col = 0; col < SCALE; col++)
                {
                    int safe_x = std::min(x * SCALE + col, source_width - 1);
                    for (int channel = 0; channel < CHANNELS; channel++)
                    {
                        reduction_area[row][col * CHANNELS + channel] =
                            (float)(input_image[safe_y * source_step + safe_x * CHANNELS + channel]);
                    }
                }
            }
            // Apply reductions iteratively
            int current_scale = SCALE;
            while (current_scale > 1) {
                for (int row = 0; row < current_scale; row++)
                    for (int col = 0; col < current_scale; col++)
                        for (int channel = 0; channel < CHANNELS; channel++)
                        {
                            float average =
                                reduction_area[row][col * CHANNELS + channel] +
                                    reduction_area[row][(col + 1) * CHANNELS + channel] +
                                    reduction_area[row + 1][col * CHANNELS + channel] +
                                    reduction_area[row + 1][(col + 1) * CHANNELS + channel];
                            reduction_area[row][col * CHANNELS + channel] =
                                (unsigned char) std::min(255.0f, std::max(0.0f, (0.25f * average)));
                        }
                current_scale /= 2;
            }
            // Copy result to output
            for (int channel = 0; channel < CHANNELS; channel++)
                result[y * target_step + x * CHANNELS + channel] = reduction_area[0][channel];
        }

}

void
resize_uchar_cpu(unsigned char *const input_image,
                 const int width,
                 const int height,
                 const int scale,
                 unsigned char *result)
{

    const int target_width = width / scale;
    const int target_height = height / scale;

    if (scale == 2)
    {
        downscale_2_uchar<3>(
            input_image, width, height, target_width, target_height, result);
    }
    else if (scale == 4)
    {
        downscale_level_uchar<3, 4>(
            input_image, width, height, target_width, target_height, result);
    }
    else if (scale == 8)
    {
        downscale_level_uchar<3, 8>(
            input_image, width, height, target_width, target_height, result);
    }

}
