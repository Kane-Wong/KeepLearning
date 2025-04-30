#include "softmax.h"

void softmax_cpu_fun0(float *input, float *output)
{
    float* row_in = input;
    float* row_out = output;
    for(int i=0; i<M; i++)
    {
        float max_value = *row_in;
        float sum_row = 0;
        for(int j=1; j<N; j++)
        {
            max_value = std::max(max_value, row_in[j]);
        }
        for(int j=0; j<N; j++)
        {
            row_out[j] = exp(row_in[j] - max_value);
            sum_row += row_out[j];
        }
        for(int j=0; j<N; j++)
        {
            row_out[j] /= sum_row;
        }
        row_in += N;
        row_out += N;
    }
}

void softmax_cpu_fun1(float* input, float* output) {
    for (int m = 0; m < M; ++m) {
        float maxval = -INFINITY;
        const float* x = input + m * N;
        for (int n = 0; n < N; ++n) {
            maxval = maxval > x[n] ? maxval : x[n];
        }
        float sumval = 0.0f;
        for (int n = 0; n < N; ++n) {
            sumval += exp(x[n] - maxval);
        }
        float* y = output + m * N;
        for (int n = 0; n < N; ++n) {
            y[n] = exp(x[n] - maxval) / sumval;
        }
    }
}

void softmax_cpu_fun2(float *input, float *output)
{
    float* row_in = input;
    float* row_out = output;
    for(int i=0; i<M; i++)
    {
        float max_value = *row_in;
        float max_value_pre = max_value;
        float sum_row = 0;
        for(int j=1; j<N; j++)
        {
            max_value_pre = max_value;
            max_value = std::max(max_value, row_in[j]);
            sum_row = sum_row * exp(max_value_pre - max_value) + exp(row_in[j] - max_value);
        }
        for(int j=0; j<N; j++)
        {
            row_out[j] = exp(row_in[j] - max_value) / sum_row;
        }
        row_in += N;
        row_out += N;
    }
}

void softmax_cpu_vector(float *input, float *output, int elem_nums)
{
    float max_value = *input;
    float sum_row = 0;
    for(int j=1; j<elem_nums; j++)
    {
        max_value = std::max(max_value, input[j]);
    }
    for(int j=0; j<elem_nums; j++)
    {
        output[j] = exp(input[j] - max_value);
        sum_row += output[j];
    }
    for(int j=0; j<elem_nums; j++)
    {
        output[j] /= sum_row;
    }

}