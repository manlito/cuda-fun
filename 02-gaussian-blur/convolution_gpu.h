
void scale(float *const input_image,
           const int width,
           const int height,
           const int channels,
           float* result);

void blur_separable_gpu(float *const input_image,
                        const int width,
                        const int height,
                        const int channels,
                        float* result);
