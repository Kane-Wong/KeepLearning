#include "attention.cuh"

int main(int argc, char **argv)
{
    const int cpu_case = atoi(argv[1]);
    const int gpu_case = atoi(argv[2]);
    printf("cpu running case %d, gpu running case %d\n", cpu_case, gpu_case);
    
    const int deviceID = 0;
    cudaSetDevice(deviceID);
    printDeviceInfor(deviceID);
    
    // 1. 环境准备
    const int q_elem_nums = batch_size * num_heads * q_token_nums * head_size;
    const int kv_elem_nums = batch_size * num_heads * kv_token_nums * head_size;
    const int q_size = q_elem_nums * sizeof(float);
    const int kv_size = kv_elem_nums * sizeof(float);
    float *host_Q = (float *)malloc(q_size);
    float *host_K = (float *)malloc(kv_size);
    float *host_V = (float *)malloc(kv_size);
    float *result_cpu = (float *)malloc(q_size);
    float *result_gpu = (float *)malloc(q_size);
    float *device_Q, *device_K, *device_V, *device_R;
    cudaMalloc((void **) &device_Q, q_size);
    cudaMalloc((void **) &device_K, kv_size);
    cudaMalloc((void **) &device_V, kv_size);
    cudaMalloc((void **) &device_R, q_size);
    for(int i=0; i<q_elem_nums; i++){
        host_Q[i] = rand() % 100;
    }
    for(int i=0; i<kv_elem_nums; i++){
        host_K[i] = rand() % 100;
        host_V[i] = rand() % 100;
    }
    
    cudaMemcpy(device_Q, host_Q, q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_K, host_K, kv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_V, host_V, kv_size, cudaMemcpyHostToDevice);

    // 2. gpu 计算
    timerecord start, end;
    start.freshtime();
    switch (gpu_case)
    {
    case 0:
        float *device_attIn, *device_attOut;
        cudaMalloc((void **) &device_attIn, batch_size * num_heads * q_token_nums * kv_token_nums * sizeof(float));
        cudaMalloc((void **) &device_attOut, batch_size * num_heads * q_token_nums * kv_token_nums * sizeof(float));
        attention_cuda_fun0<<<dim3(batch_size*num_heads), dim3(q_token_nums)>>>(device_Q, device_K, device_V, device_R, device_attIn, device_attOut);
        cudaDeviceSynchronize();
        cudaFreeHost(device_attIn);
        cudaFreeHost(device_attOut);
        break;
    default:
        printf("Error: Invalid gpu running case: %d\n", gpu_case);
        return EXIT_FAILURE;
    }
    cudaMemcpy(result_gpu, device_R, q_size, cudaMemcpyDeviceToHost);
    end.freshtime();
    printTimeGap(start, end, "gpu");
    
    // 3. cpu 计算
    start.freshtime();
    switch (cpu_case)
    {
    case 0:
        attention_cpu_fun0(host_Q, host_K, host_V, result_cpu);
        break;
    default:
        printf("Error: Invalid cpu running case: %d\n", cpu_case);
        return EXIT_FAILURE;
    }
    end.freshtime();
    printTimeGap(start, end, "cpu");

    // 4. 结果验证
    validResult<float>(result_cpu, result_gpu, q_elem_nums, 1e-4);

    // 5. 环境释放
    free(host_Q);
    free(host_K);
    free(host_V);
    free(result_cpu);
    free(result_gpu);
    cudaFreeHost(device_Q);
    cudaFreeHost(device_K);
    cudaFreeHost(device_V);
    cudaFreeHost(device_R);
   
    return 0;
}