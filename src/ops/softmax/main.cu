#include "softmax.cuh"

int main(int argc, char **argv)
{
    int cpu_case = atoi(argv[1]);
    int gpu_case = atoi(argv[2]);
    printf("cpu running case %d, gpu running case %d\n", cpu_case, gpu_case);

    int deviceID = 0;
    cudaSetDevice(deviceID);
    printDeviceInfor(deviceID);

    // 1. 环境准备
    int data_size = M * N * sizeof(float);
    float* input_host = (float *) malloc(data_size);
    float* output_host_cpu = (float *) malloc(data_size);
    float* output_host_cuda = (float *) malloc(data_size);
    for(int i=0; i<M*N; i++){
        input_host[i] = rand()%10;
    }
    float *d_a, *d_r;
    cudaMalloc((void **)&d_a, data_size);
    cudaMalloc((void **)&d_r, data_size);
    cudaMemcpy(d_a, input_host, data_size, cudaMemcpyHostToDevice);

    // 2. gpu 计算
    timerecord start, end;
    start.freshtime();
    switch (gpu_case)
    {
    case 0:
        softmax_gpu_fun0<<<dim3(1), dim3(N)>>>(d_a, d_r);
        break;
    case 1:
        softmax_gpu_fun0<<<dim3(M), dim3(1)>>>(d_a, d_r);
        break;
    case 2:
        softmax_gpu_fun1<N><<<dim3(M), dim3(N)>>>(d_a, d_r);
        break;
    case 3:
        softmax_gpu_fun2<<<dim3(M), dim3(32)>>>(d_a, d_r, M, N);
        break;
    case 4:
        softmax_gpu_fun3<<<dim3(M), dim3(1)>>>(d_a, d_r);
        break;
    case 5:
        softmax_gpu_fun4<<<dim3(M), dim3(32)>>>(d_a, d_r, M, N);
        break;
    default:
        printf("Error: Invalid gpu running case: %d\n", gpu_case);
        return EXIT_FAILURE;
    }
    cudaDeviceSynchronize(); 
    cudaMemcpy(output_host_cuda, d_r, data_size, cudaMemcpyDeviceToHost);
    end.freshtime();
    printTimeGap(start, end, "gpu");
    
    // 3. cpu 计算
    start.freshtime();
    switch (cpu_case)
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
        printf("Error: Invalid cpu running case: %d\n", cpu_case);
        return EXIT_FAILURE;
    }
    end.freshtime();
    printTimeGap(start, end, "cpu");

    // 4. 结果验证
    validResult<float>(output_host_cpu, output_host_cuda, M*N, 1e-4);

    // 5. 环境释放
    free(input_host);
    free(output_host_cpu);
    free(output_host_cuda);
    cudaFreeHost(d_a);
    cudaFreeHost(d_r);
   
    return 0;
}