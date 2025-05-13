#include "attention.h"

// 一个block处理一个head, block内每个线程处理一个q_token。
__global__ void attention_cuda_fun0(float* matQ, float* matK, float* matV, float* matR, float* attIn, float *attOut);