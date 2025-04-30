#include "reduce_sum.cuh"

int main(int argc, char **argv){
    int cpu_case = atoi(argv[1]);
    int gpu_case = atoi(argv[2]);
    printf("cpu running case %d, gpu running case %d\n", cpu_case, gpu_case);

    int deviceID = 0;
    cudaSetDevice(deviceID);
    printDeviceInfor(deviceID);
    
    // 1. 环境准备
    const int vector_size = 1024;
    size_t data_size = vector_size * vector_size * sizeof(float);
    size_t result_size = vector_size * sizeof(float);
    float *a = (float *)malloc(data_size);
    float *r_c = (float *)malloc(result_size);
    float *r_g = (float *)malloc(result_size);

    for (int i=0; i<vector_size*vector_size; i++){
        a[i] = i%10;
    }
    for (int i=0; i<vector_size; i++){
        r_c[i] = 0;
        r_g[i] = 0;
    }
    
    float *d_a, *d_r;
    cudaMalloc((void **)&d_a, data_size);
    cudaMalloc((void **)&d_r, result_size);
    cudaMemcpy(d_a, a, data_size, cudaMemcpyHostToDevice);

    // 2. gpu 计算
    timerecord start, end;
    start.freshtime();
    switch (gpu_case)
    {
    case 0:
        reduce_kernel_fun0<<<dim3(vector_size), dim3(vector_size)>>>(d_a, d_r);
        break;
    case 1:
        reduce_kernel_fun1<vector_size><<<dim3(vector_size), dim3(vector_size)>>>(d_a, d_r);
        break;
    case 2:
        reduce_kernel_fun2<vector_size><<<dim3(vector_size), dim3(vector_size)>>>(d_a, d_r);
        break;
    case 3:
        reduce_kernel_fun3<vector_size><<<dim3(vector_size), dim3(vector_size)>>>(d_a, d_r);
        break;
    case 4:
        reduce_kernel_fun4<vector_size/2><<<dim3(vector_size), dim3(vector_size/2)>>>(d_a, d_r);
        break;
    case 5:
        reduce_kernel_fun5<vector_size/2><<<dim3(vector_size), dim3(vector_size/2)>>>(d_a, d_r);
        break;
    case 6:
        reduce_kernel_fun6<<<dim3(vector_size), dim3(32)>>>(d_a, d_r, vector_size, vector_size);
        break;
    default:
        printf("Error: Invalid gpu running case: %d\n", gpu_case);
        return EXIT_FAILURE;
    }    
    cudaDeviceSynchronize(); 
    cudaMemcpy(r_g, d_r, result_size, cudaMemcpyDeviceToHost);
    end.freshtime();
    printTimeGap(start, end, "gpu");

    // 3. cpu 计算
    start.freshtime();
    switch (cpu_case)
    {
    case 0:
        _cpu_compute(a, r_c, vector_size);
        break;
    default:
        printf("Error: Invalid cpu running case: %d\n", cpu_case);
        return EXIT_FAILURE;
    }
    end.freshtime();
    printTimeGap(start, end, "cpu");
    
    // 4. 结果验证
    validResult<float>(r_c, r_g, vector_size, 1e-10);

    // 5. 环境释放
    free(a);
    free(r_c);
    free(r_g);
    cudaFreeHost(d_a);
    cudaFreeHost(d_r);
    return 0;
}