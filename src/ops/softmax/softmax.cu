#include "softmax.cuh"

// 开启M个线程，对应M行
__global__ void softmax_gpu_fun0(float* input, float* output)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float* row_in = input + tid * N;
    float* row_out = output + tid * N;
    float max_value = *row_in;
    float sum_row = 0;
    if(tid<M)
    {
        for(int j=1; j<N; j++)
        {
            max_value = max(max_value, row_in[j]);
        }
        for(int j=0; j<N; j++)
        {
            sum_row +=  exp(row_in[j] - max_value);
        }
        for(int j=0; j<N; j++)
        {
            row_out[j] = exp(row_in[j] - max_value) / sum_row;
        }
    }
}

// 较fun1 可支持更大的横轴点数，
__global__ void softmax_gpu_fun2(float* input, float* output, const int M_, const int N_) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    assert(blockDim.x == warpSize); 
    for (int i = bid; i < M_; i += gridDim.x) {
        const float* row_pointer_in = input + i * N_;
        float* row_pointer_out = output + i * N_;

        float maxval = -INFINITY;
        for (int j = tid; j < N_; j += warpSize) {
            maxval = max(maxval, row_pointer_in[j]);
        }
        // warp-reduce to calculate the MAX of maxval among all lanes
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            maxval = max(maxval, __shfl_xor_sync(0xFFFFFFFF, maxval, offset));
        }

        float sumval = 0.0f;
        for (int j = tid; j < N_; j += warpSize) {
            sumval += exp(row_pointer_in[j] - maxval);
        }
        // warp-reduce to calculate the SUM of sumval among all lanes
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sumval += __shfl_xor_sync(0xFFFFFFFF, sumval, offset);
        }
        
        for (int j = tid; j < N_; j += warpSize) {
            row_pointer_out[j] = exp(row_pointer_in[j] - maxval) / sumval ;
        }
    }
}

// fun0 online softmax
__global__ void softmax_gpu_fun3(float* input, float* output)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float* row_in = input + tid * N;
    float* row_out = output + tid * N;
    float max_value = *row_in;
    float max_value_pre = max_value;
    float sum_row = 0;
    if(tid<M)
    {
        for(int j=1; j<N; j++)
        {
            max_value_pre = max_value;
            max_value = max(max_value, row_in[j]);
            sum_row = sum_row * exp(max_value_pre - max_value) + exp(row_in[j] - max_value);
        }
        for(int j=0; j<N; j++)
        {
            row_out[j] = exp(row_in[j] - max_value) / sum_row;
        }
    }
}

// fun2 online softmax
__global__ void softmax_gpu_fun4(float* input, float* output, const int M_, const int N_) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    assert(blockDim.x == warpSize); 
    for (int i = bid; i < M_; i += gridDim.x) {
        const float* row_pointer_in = input + i * N_;
        float* row_pointer_out = output + i * N_;

        float maxval = -INFINITY, bigger;
        float sumval = 0.0f;
        for (int j = tid; j < N_; j += warpSize) {
            bigger  = max(maxval, row_pointer_in[j]);
            sumval = sumval * exp(maxval - bigger) + exp(row_pointer_in[j] - bigger); 
            maxval = bigger;
        }
        
        // warp-reduce to calculate the MAX of maxval among all lanes
        // #pragma unroll
        float offsetMax, offsetSum;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            __syncwarp();
            offsetMax = __shfl_xor_sync(0xFFFFFFFF, maxval, offset);
            offsetSum = __shfl_xor_sync(0xFFFFFFFF, sumval , offset);
            if(offsetMax > maxval)
            {
                sumval *= exp(maxval - offsetMax);
                maxval = offsetMax;
            }
            else
            {
                offsetSum *= exp(offsetMax - maxval);
            }
            sumval += offsetSum;
        }
        
        for (int j = tid; j < N_; j += warpSize) {
            row_pointer_out[j] = exp(row_pointer_in[j] - maxval) / sumval ;
        }
    }
}