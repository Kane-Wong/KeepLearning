#include "common.h"
#pragma once

constexpr int num_heads  = 12;
constexpr int token_nums = 128;
constexpr int batch_size  = 1;
constexpr int hidden_size = 768;
constexpr int head_size = hidden_size / num_heads;
constexpr int q_token_nums = token_nums;
constexpr int kv_token_nums = token_nums;

void attention_cpu_fun0(float* matQ, float* matK, float* matV, float* matR);
