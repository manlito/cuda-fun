cmake_minimum_required(VERSION 3.0)

project(cuda_fun_resize)
enable_language(CUDA)

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

# Default build type: Release
if (NOT CMAKE_BUILD_TYPE AND NOT MSVC)
    set(CMAKE_BUILD_TYPE "Release")
endif ()


find_package(JPEG REQUIRED)

add_executable(cuda_fun_resize
        main.cu
        resize.cu
        image.cpp
        )

add_executable(cuda_fun_resize_from_dir
        main-from-dir.cu
        resize.cu
        image.cpp
        )

add_executable(cuda_fun_resize_streams
        main-from-dir-streams.cu
        resize.cu
        image.cpp
        )

add_executable(cuda_fun_resize_uchar
        main-uchar.cu
        resize-uchar.cu
        resize.cu
        image.cu
        image.cpp
        )

add_executable(cuda_fun_resize_streams_uchar
        main-from-dir-streams-uchar.cu
        resize-uchar.cu
        image.cu
        image.cpp
        )

include_directories(${JPEG_INCLUDE_DIRS})
target_link_libraries(cuda_fun_resize ${JPEG_LIBRARIES})
target_link_libraries(cuda_fun_resize_from_dir ${JPEG_LIBRARIES})
target_link_libraries(cuda_fun_resize_streams ${JPEG_LIBRARIES})
target_link_libraries(cuda_fun_resize_uchar ${JPEG_LIBRARIES})
target_link_libraries(cuda_fun_resize_streams_uchar ${JPEG_LIBRARIES})

install(TARGETS cuda_fun_resize RUNTIME DESTINATION bin)
