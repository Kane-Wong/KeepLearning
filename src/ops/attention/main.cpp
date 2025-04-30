#include "attention.h"

int main(int argc, char **argv)
{
    assert(argc == 2);
    const int case_number = atoi(argv[1]);
    printf("cpu running case %d\n", case_number);
    
    // 1. 环境准备
    const int elem_nums = batch_size * num_heads * token_nums * head_size;
    const int data_size = elem_nums * sizeof(float);
    float *host_Q = (float *)malloc(data_size);
    float *host_K = (float *)malloc(data_size);
    float *host_V = (float *)malloc(data_size);
    float *result_cpu = (float *)malloc(data_size);

    for(int i=0; i<elem_nums; i++){
        host_Q[i] = rand() % 100;
        host_K[i] = rand() % 100;
        host_V[i] = rand() % 100;
    }
    
    // 2. cpu 计算
    timerecord start, end;
    start.freshtime();
    switch (case_number)
    {
    case 0:
        attention_cpu_fun0(host_Q, host_K, host_V, result_cpu);
        break;
    default:
        printf("Error: Invalid cpu running case: %d\n", case_number);
        return EXIT_FAILURE;
    }
    end.freshtime();
    printTimeGap(start, end, "cpu");

    // 3. 环境释放
    free(host_Q);
    free(host_K);
    free(host_V);
    free(result_cpu);
    return 0;
}