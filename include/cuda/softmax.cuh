#include "softmax.h"

// 开启M个线程，每个线程处理N个元素的softmax
__global__ void softmax_gpu_fun0(float* input, float* output, const int M, const int N);

// 开启 M*N 个线程，归约求 max 和 sum， 最大支持横轴点数：1024
template<int BLOCK_SIZE>
__global__ void softmax_gpu_fun1(float* input, float* output)
{
    __shared__ float sdata[BLOCK_SIZE];     // block内共享
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];  
    __syncthreads();

    // 求每行最大值, 最终保存在sdata[0]
    // float max_value = -INFINITY;
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    // if (tid == 0) max_value = sdata[0];

    // 求指数
    sdata[tid] = exp(input[i] - sdata[0]);
    output[i] = sdata[tid];
    __syncthreads();

    // 指数求和, 最终保存在sdata[0]
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    output[i] /= sdata[0];
}

// 较fun1 可支持更大的横轴点数，
__global__ void softmax_gpu_fun2(float* input, float* output, const int M_, const int N_);

// fun0 online softmax
__global__ void softmax_gpu_fun3(float* input, float* output, const int M, const int N);

// fun2 online softmax
__global__ void softmax_gpu_fun4(float* input, float* output, const int M_, const int N_);

// 对每一行做softmax时调用
__device__ inline void softmax_vector(float *input, float *output, int elem_nums)
{
    float max_value = *input;
    float sum_row = 0;
    for(int j=1; j<elem_nums; j++)
    {
        max_value = fmaxf(max_value, input[j]);
    }
    for(int j=0; j<elem_nums; j++)
    {
        output[j] = expf(input[j] - max_value);
        sum_row += output[j];
    }
    for(int j=0; j<elem_nums; j++)
    {
        output[j] /= sum_row;
    }

}