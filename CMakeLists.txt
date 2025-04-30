cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
Project(CUDA_KERNELS CXX C CUDA)
set(CMAKE_CUDA_ARCHITECTURES 70 86) # titanv_Volta 70  rtx30_Ampere 86
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_compile_options(-O3 -g -Wall)
include_directories(include)
include_directories(include/cpu)
include_directories(include/cuda)

add_subdirectory(src/ops/matmul)
add_subdirectory(src/ops/attention)
add_subdirectory(src/ops/reduce)
add_subdirectory(src/ops/softmax)