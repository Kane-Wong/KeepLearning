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
    const int elem_nums = batch_size * num_heads * token_nums * head_size;
    const int data_size = elem_nums * sizeof(float);
    float *host_Q = (float *)malloc(data_size);
    float *host_K = (float *)malloc(data_size);
    float *host_V = (float *)malloc(data_size);
    float *result_cpu = (float *)malloc(data_size);
    float *result_gpu = (float *)malloc(data_size);
    float *device_Q, *device_K, *device_V, *device_R;
    cudaMalloc((void **) &device_Q, data_size);
    cudaMalloc((void **) &device_K, data_size);
    cudaMalloc((void **) &device_V, data_size);
    cudaMalloc((void **) &device_R, data_size);
    
    for(int i=0; i<elem_nums; i++){
        host_Q[i] = rand() % 100;
        host_K[i] = rand() % 100;
        host_V[i] = rand() % 100;
    }
    cudaMemcpy(device_Q, host_Q, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_K, host_K, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_V, host_V, data_size, cudaMemcpyHostToDevice);

    // 2. gpu 计算
    timerecord start, end;
    start.freshtime();
    switch (gpu_case)
    {
    case 0:
        _cuda_compute_fun0<<<dimGrid, dimBlock>>>(d_a, d_b, d_r, matrix_size);
        break;
    default:
        printf("Error: Invalid gpu running case: %d\n", gpu_case);
        return EXIT_FAILURE;
    }
    cudaDeviceSynchronize(); // 同步，且检查执行期间发生的错误
    cudaMemcpy(result_gpu, device_R, data_size, cudaMemcpyDeviceToHost);
    end.freshtime();
    printTimeGap(start, end, "gpu");
    
    // 3. cpu 计算
    start.freshtime();
    switch (cpu_case)
    {
    case 0:
        _cpu_compute_fun0(a, b, r, matrix_size);
        break;
    default:
        printf("Error: Invalid cpu running case: %d\n", cpu_case);
        return EXIT_FAILURE;
    }
    end.freshtime();
    printTimeGap(start, end, "cpu");

    // 4. 结果验证
    validResult<float>(r, r_re, matrix_size*matrix_size, 1e-10);

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