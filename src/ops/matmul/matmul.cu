#include "matmul.cuh"

// 每次计算的读/写均从全局内存取数据
__global__ void _cuda_compute_fun0(float *a, float *b, float *r, int matrix_size){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    // printf("%d, %d, %d, %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
    if (x<matrix_size && y<matrix_size){
        for (int k=0; k<matrix_size; k++){
            r[y*matrix_size+x] += a[y*matrix_size+k] * b[k*matrix_size+x];
        }
    }
}

// 每次循环计算的sum作为局部变量存在寄存器中
__global__ void _cuda_compute_fun1(float *a, float *b, float *r, int matrix_size){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x<matrix_size && y<matrix_size){
        float sum=0.0;
        for (int k=0; k<matrix_size; k++){
            sum += a[y*matrix_size+k] * b[k*matrix_size+x];
        }
        r[y*matrix_size+x] = sum;
    }
}
