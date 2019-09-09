cmake_minimum_required(VERSION 3.0)

project(cuda_fun_vector_add)
enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Default build type: Release
if (NOT CMAKE_BUILD_TYPE AND NOT MSVC)
  set(CMAKE_BUILD_TYPE "Release")
endif ()


add_executable(cuda_fun_vector_add main.cu)

install(TARGETS cuda_fun_vector_add RUNTIME DESTINATION bin)
