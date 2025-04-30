#include "common.h"

void _cpu_compute(float *input, float *out, int token_num);

__global__ void reduce_kernel_fun0(float *input, float *output);

template <int BLOCK_SIZE>
__global__ void reduce_kernel_fun1(float *input, float *output)
{
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

template <int BLOCK_SIZE>
__global__ void reduce_kernel_fun2(float *input, float *output)
{
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid * 2 * s < blockDim.x)
        {
            int index = tid * 2 * s; 
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

template <int BLOCK_SIZE>
__global__ void reduce_kernel_fun3(float *input, float *output)
{
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

// 计算与访存并行， BLOCK_SIZE 为elements numbers的一半
template <int BLOCK_SIZE>
__global__ void reduce_kernel_fun4(float *input, float *output)
{
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    sdata[tid] = input[i] + input[i + blockDim.x];      
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

template <int BLOCK_SIZE>
__global__ void reduce_kernel_fun5(float *input, float *output)
{
    volatile __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    sdata[tid] = input[i] + input[i + blockDim.x];      
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 32; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32)
    {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

__global__ void reduce_kernel_fun6(float *input, float *output, const int M, const int N);