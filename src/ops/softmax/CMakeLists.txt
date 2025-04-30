cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
Project(CUDA_KERNELS CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-O3 --use_fast_math -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_ARCHITECTURES 70 86) # titanv_Volta 70  rtx30_Ampere 86
if(NOT DEFINED EXECUTABLE_OUTPUT_PATH)
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
endif()

add_compile_options(-O3 -g -Wall)
include_directories(../../../include)
include_directories(../../../include/cpu)

set(SOURCE_CPU
    softmax.cpp
    ../../common.cpp 
)
# add_executable(softmax_cpu ${SOURCE_CPU} main.cpp)

set(SOURCE_GPU
    softmax.cu
    ../../common.cu 
)
include_directories(../../../include/cuda)
add_executable(softmax_gpu ${SOURCE_CPU} ${SOURCE_GPU} main.cu)