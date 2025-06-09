#include "common.h"
#pragma once

// 对一维向量求 softmax
void softmax_cpu_vector(float *input, float *output, int elem_nums);

void softmax_cpu_fun0(float *input, float *output, const int M, const int N);

void softmax_cpu_fun1(float* input, float* output, const int M, const int N);

void softmax_cpu_fun2(float *input, float *output, const int M, const int N);