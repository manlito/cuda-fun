/**
 * Modified version from
 * https://github.com/simonfuhrmann/mve/blob/master/libs/mve/image_io.cc
 *
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <jpeglib.h>

void
jpg_error_handler(j_common_ptr /*cinfo*/);

void
jpg_message_handler(j_common_ptr /*cinfo*/, int msg_level);

void
read_jpeg_image_cu(const std::string &filename,
                   int &width,
                   int &height,
                   int &channels,
                   int &current_allocation_size,
                   unsigned char **image_data,
                   void *stream = nullptr);

void
read_jpeg_image(const std::string &filename,
                std::vector<unsigned char> &image_data,
                int &width,
                int &height,
                int &channels);

void
write_jpeg_image(const std::string &filename,
                 const unsigned char *const image_data,
                 const int &width,
                 const int &height,
                 const int channels);
