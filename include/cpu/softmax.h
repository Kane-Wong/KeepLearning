#include "common.h"
#pragma once

constexpr int M  = 1024;
constexpr int N  = 1024;

// 对一维向量求 softmax
void softmax_cpu_vector(float *input, float *output, int elem_nums);

void softmax_cpu_fun0(float *input, float *output);

void softmax_cpu_fun1(float* input, float* output);

void softmax_cpu_fun2(float *input, float *output);