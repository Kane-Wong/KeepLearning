#include "reduce_sum.cuh"

void _cpu_compute(float *input, float *out, int token_num)
{
    float *anc;
    for (int i = 0; i < token_num; i++)
    {
        anc = input + i * token_num;
        // std::cout << *anc << ' ' << i << std::endl;
        for (int j = 0; j < token_num; j++)
        {
            out[i] += anc[j];
        }
    }
}

__global__ void reduce_kernel_fun0(float *input, float *output)
{
    int tid = threadIdx.x;
    float *x = input + blockDim.x * blockIdx.x;
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            // if(0 == blockIdx.x){
            //     printf("tid: %d, tid_s: %d, a: %f, b: %f \n", tid, tid+s, x[tid], x[tid+s]);
            // }
            x[tid] += x[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
        output[blockIdx.x] = x[0];
}


__global__ void reduce_kernel_fun6(float *input, float *output, const int M, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    for(int i = bid; i < M; i += gridDim.x){
        float* row_pointer = input + i * N;
        float sum_val = 0;
        for(int j = tid; j < N; j += warpSize)
        {
            sum_val += row_pointer[j];
        }
        for(int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            sum_val += __shfl_xor_sync(0xffffffff, sum_val, offset);
        }
        output[i] = sum_val;
    }
}