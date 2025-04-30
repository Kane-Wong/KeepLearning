#include "softmax.h"

int main(int argc, char **argv)
{
    assert(argc == 2);
    const int case_number = atoi(argv[1]);
    printf("cpu running case %d\n", case_number);

    // 1. 环境准备
    int data_size = M * N * sizeof(float);
    float* input_host = (float *) malloc(data_size);
    float* output_host_cpu = (float *) malloc(data_size);
    for(int i=0; i<M*N; i++){
        input_host[i] = rand()%10;
    }

    // 2. cpu 计算
    timerecord start, end;
    start.freshtime();
    switch (case_number)
    {
    case 0:
        softmax_cpu_fun0(input_host, output_host_cpu);
        break;
    case 1:
        softmax_cpu_fun1(input_host, output_host_cpu);
        break;
    case 2:
        softmax_cpu_fun2(input_host, output_host_cpu);
        break;
    default:
        printf("Error: Invalid cpu running case: %d\n", case_number);
        return EXIT_FAILURE;
    }
    end.freshtime();
    printTimeGap(start, end, "cpu");

    // 3. 环境释放
    free(input_host);
    free(output_host_cpu);
   
    return 0;
}