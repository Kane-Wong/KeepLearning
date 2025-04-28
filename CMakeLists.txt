cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
Project(CUDA_KERNELS CXX C CUDA)
# set(CMAKE_CUDA_FLAGS "-O3 --use_fast_math -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_ARCHITECTURES 70 86) # titanv_Volta 70  rtx30_Ampere 86
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(./include)

# add_subdirectory(ops/softmax)
# add_subdirectory(ops/reduce)
add_subdirectory(ops/matmul)