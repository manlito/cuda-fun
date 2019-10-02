/**
 * Modified version from
 * https://github.com/simonfuhrmann/mve/blob/master/libs/mve/image_io.cc
 *
 */

#pragma once

#include <fstream>
#include <jpeglib.h>
#include <cstring>
#include <iostream>
#include <exception>
#include <vector>

void
jpg_error_handler (j_common_ptr /*cinfo*/) {
    throw std::runtime_error("JPEG format not recognized");
}

void
jpg_message_handler (j_common_ptr /*cinfo*/, int msg_level) {
    if (msg_level < 0)
        throw std::runtime_error("JPEG data corrupt");
}

void read_jpeg_image(const std::string &filename,
                     std::vector<unsigned char> &image_data,
                     int &width,
                     int &height,
                     int &channels) {

    FILE* fp = std::fopen(filename.c_str(), "rb");
    if (fp == nullptr)
        throw std::runtime_error(std::strerror(errno));

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;
    try {
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
        image_data.resize(width * height * channels);

        /* Start decompression. */
        jpeg_start_decompress(&cinfo);

        unsigned char* data_ptr = &image_data[0];
        while (cinfo.output_scanline < cinfo.output_height) {
            jpeg_read_scanlines(&cinfo, &data_ptr, 1);
            data_ptr += channels * cinfo.output_width;
        }

        /* Shutdown JPEG decompression. */
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        std::fclose(fp);
    }
    catch (...) {
        jpeg_destroy_decompress(&cinfo);
        std::fclose(fp);
        throw std::runtime_error("Failed to open JPG image");
    }

}

void write_jpeg_image(const std::string &filename,
                      const std::vector<unsigned char> &image_data,
                      const int &width,
                      const int &height,
                      const int &channels) {

    if (channels != 1 && channels != 3)
        throw std::runtime_error("Invalid image color space");

    FILE* fp = std::fopen(filename.c_str(), "wb");
    if (!fp)
        throw std::runtime_error(std::strerror(errno));

    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;

    /* Setup error handler and info object. */
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, fp);

    /* Specify image dimensions. */
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = channels;
    cinfo.in_color_space = (channels == 1 ? JCS_GRAYSCALE : JCS_RGB);

    /* Set default compression parameters. */
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 90, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    const unsigned char* data = &image_data[0];
    int row_stride = width * channels;
    while (cinfo.next_scanline < cinfo.image_height)
    {
        JSAMPROW row_pointer = const_cast<JSAMPROW>(&data[cinfo.next_scanline * row_stride]);
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    std::fclose(fp);
}
