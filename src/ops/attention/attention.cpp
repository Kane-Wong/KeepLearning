#include "attention.h"
#include "softmax.h"

void attention_cpu_fun0(float* matQ, float* matK, float* matV, float* matR)
{
    float *att_vec = (float *)malloc(kv_token_nums * sizeof(float)); //save attention vector each row; 
    float *softmax_vec = (float *)malloc(kv_token_nums * sizeof(float)); 
    float *q_start = matQ;
    float *k_start = matK;
    float *v_start = matV;
    float *r_start = matR;
    float *k_temp = k_start;
    float *v_temp = v_start;

    for(int b=0; b<batch_size; b++)
    {
        for(int h=0; h<num_heads; h++)
        {
            for(int t=0; t<q_token_nums; t++)
            {
                k_temp = k_start;
                for(int k=0; k<kv_token_nums; k++)
                {
                    float sum_val = 0;
                    for (int i=0; i<head_size; i++)
                    {
                        sum_val += q_start[i] * k_temp[i];
                    }
                    att_vec[k] = sum_val / sqrt(head_size);
                    k_temp += head_size;
                }
                softmax_cpu_vector(att_vec, softmax_vec, kv_token_nums);

                for(int i=0; i<head_size; i++)
                {
                    float sum_val = 0;
                    
                    v_temp = v_start + i;
                    for(int k=0; k<kv_token_nums; k++)
                    {
                        sum_val += softmax_vec[k] * (*v_temp);
                        v_temp += head_size;
                    }
                    *r_start = sum_val;
                    r_start += 1;
                    
                }
                q_start += head_size;

            }
            k_start += kv_token_nums * head_size;
            v_start += kv_token_nums * head_size;
        }
    }
    free(att_vec);
    free(softmax_vec);
}