#include "matmul.cuh"

int main(int argc, char **argv)
{
    int cpu_case = atoi(argv[1]);
    int gpu_case = atoi(argv[2]);
    printf("cpu running case %d, gpu running case %d\n", cpu_case, gpu_case);
    
    int deviceID = 0;
    cudaSetDevice(deviceID);
    printDeviceInfor(deviceID);
    
    // 1. 环境准备
    size_t size = matrix_size * matrix_size * sizeof(float);
    float *a = (float *)malloc(size);
    float *b = (float *)malloc(size);
    float *r = (float *)malloc(size);
    float *r_re = (float *)malloc(size);
    for(int i=0; i<matrix_size*matrix_size; i++){
        a[i] = rand() % matrix_size;
        b[i] = rand() % matrix_size;
        r[i] = 0;
        r_re[i] = 0;
    }

    float *d_a, *d_b, *d_r;
    cudaMalloc((void **) &d_a, size);
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_r, size);
    const size_t block_x=64, block_y=64;
    size_t grid_x = (matrix_size+block_x -1) / block_x;
    size_t grid_y = (matrix_size+block_y-1) / block_y;
    dim3 dimGrid(grid_x, grid_y);
    dim3 dimBlock(block_x, block_y);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // 2. gpu 计算
    timerecord start, end;
    start.freshtime();
    switch (gpu_case)
    {
    case 0:
        _cuda_compute_fun0<<<dimGrid, dimBlock>>>(d_a, d_b, d_r, matrix_size);
        break;
    case 1:
        _cuda_compute_fun1<<<dimGrid, dimBlock>>>(d_a, d_b, d_r, matrix_size);
        break;
    case 2:
        _cuda_compute_fun2<block_x, block_y><<<dimGrid, dimBlock>>>(d_a, d_b, d_r, matrix_size);
        break;
    case 3:
        _cuda_compute_fun3<block_x, block_y><<<dimGrid, dim3(block_x/8, block_y/8)>>>(d_a, d_b, d_r, matrix_size); 
        break;
    default:
        printf("Error: Invalid gpu running case: %d\n", gpu_case);
        return EXIT_FAILURE;
    }
    cudaDeviceSynchronize(); // 同步，且检查执行期间发生的错误
    cudaMemcpy(r_re, d_r, size, cudaMemcpyDeviceToHost);
    end.freshtime();
    printTimeGap(start, end, "gpu");
    
    // 3. cpu 计算
    start.freshtime();
    switch (cpu_case)
    {
    case 0:
        _cpu_compute_fun0(a, b, r, matrix_size);
        break;
    case 1:
        _cpu_compute_fun1(a, b, r, matrix_size);
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
    free(a);
    free(b);
    free(r);
    free(r_re);
    cudaFreeHost(d_a);
    cudaFreeHost(d_b);
    cudaFreeHost(d_r);
   
    return 0;
}