cmake_minimum_required(VERSION 3.0)

project(cuda_fun_gaussian_blur)
enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Default build type: Release
if (NOT CMAKE_BUILD_TYPE AND NOT MSVC)
  set(CMAKE_BUILD_TYPE "Release")
endif ()


find_package(JPEG REQUIRED)

add_executable(cuda_fun_gaussian_blur 
    main.cu 
    convolution_gpu_blur_separable.cu
    convolution_gpu_scale.cu
    )

include_directories(${JPEG_INCLUDE_DIRS})
target_link_libraries(cuda_fun_gaussian_blur ${JPEG_LIBRARIES})

install(TARGETS cuda_fun_gaussian_blur RUNTIME DESTINATION bin)
