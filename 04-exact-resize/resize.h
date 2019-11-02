
void
resize(float *const input_image,
       const int width,
       const int height,
       const int scale,
       float *result,
       cudaStream_t *stream = nullptr);

void
resize_uchar(unsigned char *const input_image,
             const int width,
             const int height,
             const int scale,
             unsigned char *result,
             cudaStream_t *stream = nullptr);
