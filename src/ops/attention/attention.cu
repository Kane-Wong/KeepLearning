#include "attention.cuh"
#include "softmax.cuh"


// 共[batch_size*num_head]个block，每个block有num_Q个线程，每个线程处理一个q_token
__global__ void attention_cuda_fun0(float* matQ, float* matK, float* matV, float* matR, float* attIn, float *attOut)
{
    const int headID = blockIdx.x;
    const int tID = threadIdx.x;
    assert(q_token_nums<1024);  // 每个block内最多处理1024个q_token
    float *Q_start = matQ + headID * q_token_nums * head_size + tID * head_size;
    float *K_start = matK + headID * kv_token_nums * head_size;
    float *V_start = matV + headID * kv_token_nums * head_size;
    float *R_start = matR + headID * q_token_nums * head_size +  tID * head_size;
    float *attIn_start = attIn + headID * q_token_nums * kv_token_nums + tID * kv_token_nums;
    float *attOut_start = attOut + headID * q_token_nums * kv_token_nums + tID * kv_token_nums;
    float *v_temp = V_start;

    for(int k=0; k<kv_token_nums; k++)
    {
        float sum_val = 0;
        for (int i=0; i<head_size; i++)
        {
            sum_val += Q_start[i] * K_start[i];
        }
        attIn_start[k] = sum_val / sqrtf(head_size);
        K_start += head_size;
    }
    softmax_vector(attIn_start, attOut_start, kv_token_nums);

    for(int i=0; i<head_size; i++)
    {
        float sum_val = 0;
        v_temp = V_start + i;
        for(int k=0; k<kv_token_nums; k++)
        {
            sum_val += attOut_start[k] * (*v_temp);
            v_temp += head_size;
        }
        R_start[i] = sum_val; 
    }
}