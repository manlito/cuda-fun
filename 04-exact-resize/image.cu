/**
 * Modified version from
 * https://github.com/simonfuhrmann/mve/blob/master/libs/mve/image_io.cc
 *
 * This one is able to call cudaMallocManaged if the image wont fit, or the pointer
 * is not yet initialized (current allocation size of 0)
 *
 */

#include <fstream>
#include <jpeglib.h>
#include <cstring>
#include <iostream>
#include <exception>
#include <vector>
#include "image.h"
#include "error_check.h"

void
read_jpeg_image_cu(const std::string &filename,
                   int &width,
                   int &height,
                   int &channels,
                   std::function<void(const int &new_allocation)> allocation_function,
                   unsigned char **image_data)
{

    FILE *fp = std::fopen(filename.c_str(), "rb");
    if (fp == nullptr)
        throw std::runtime_error(std::strerror(errno));

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;
    try
    {
        /* Setup error handler and JPEG reader. */
        cinfo.err = jpeg_std_error(&jerr);
        jerr.error_exit = &jpg_error_handler;
        jerr.emit_message = &jpg_message_handler;
        jpeg_create_decompress(&cinfo);
        jpeg_stdio_src(&cinfo, fp);

        /* Read JPEG header. */
        int ret = jpeg_read_header(&cinfo, static_cast<boolean>(false));
        if (ret != JPEG_HEADER_OK)
            throw std::runtime_error("JPEG header not recognized");

        if (cinfo.out_color_space != JCS_GRAYSCALE
            && cinfo.out_color_space != JCS_RGB)
            throw std::runtime_error("Invalid JPEG color space");

        /* Create image. */
        width = cinfo.image_width;
        height = cinfo.image_height;
        channels = (cinfo.out_color_space == JCS_RGB ? 3 : 1);

        const int allocation_size = (width * height * channels);
        /* Allocate memory for this image */
        allocation_function(allocation_size);

        /* Start decompression. */
        jpeg_start_decompress(&cinfo);

        unsigned char* data_ptr = &(*image_data)[0];
        while (cinfo.output_scanline < cinfo.output_height)
        {
            jpeg_read_scanlines(&cinfo, &data_ptr, 1);
            data_ptr += channels * cinfo.output_width;
        }

        /* Shutdown JPEG decompression. */
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        std::fclose(fp);
    }
    catch (...)
    {
        jpeg_destroy_decompress(&cinfo);
        std::fclose(fp);
        throw std::runtime_error("Failed to open JPG image");
    }

}
