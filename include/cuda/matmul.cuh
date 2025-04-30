#include "matmul.h"


// 每次计算的读/写均从全局内存取数据
__global__ void _cuda_compute_fun0(float *a, float *b, float *r, int matrix_size);

// 每次循环计算的sum作为局部变量存在寄存器中
__global__ void _cuda_compute_fun1(float *a, float *b, float *r, int matrix_size);

// 沿K维度切成若干份，每次计算将每份数据由全局内存加载至共享内存中
template<const int BLOCK_SIZE_X, const int BLOCK_SIZE_Y>
__global__ void _cuda_compute_fun2(float *a, float *b, float *r, int matrix_size)
{
    const int blockK = 32;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    __shared__ float matA[BLOCK_SIZE_Y][blockK];
    __shared__ float matB[blockK][BLOCK_SIZE_X];
    assert(BLOCK_SIZE_X == BLOCK_SIZE_Y && BLOCK_SIZE_X == blockK);
    float sumVal = 0.0f;
    for(int i=0; i<matrix_size/blockK; i++)
    {
        // 利用线程阵列将数据从全局内存放置共享内存中，需要 blockK = BLOCK_SIZE_X = BLOCK_SIZE_Y
        matA[threadIdx.y][threadIdx.x] = *(a + y * matrix_size + i * blockK + threadIdx.x);
        matB[threadIdx.y][threadIdx.x] = *(b + (i * blockK + threadIdx.y) * matrix_size + x);
        __syncthreads();
        
        for(int j=0; j<blockK; j++)
        {
            sumVal += matA[threadIdx.y][j] * matB[j][threadIdx.x];
        }
        __syncthreads();
    }
    r[y*matrix_size+x] = sumVal;
}

template<const int BLOCK_SIZE_X, const int BLOCK_SIZE_Y>
__global__ void _cuda_compute_fun3(float *a, float *b, float *r, int matrix_size)
{
    const int blockK = 8;
    const int threadM = 8;
    const int threadN = 8;
    const int x = (blockDim.x * blockIdx.x + threadIdx.x) * threadN;
    const int y = (blockDim.y * blockIdx.y + threadIdx.y) * threadM;
    __shared__ float matA[BLOCK_SIZE_Y][blockK];
    __shared__ float matB[blockK][BLOCK_SIZE_X];
    float threaA[threadM], threaB[threadN];
    float threaR[threadM][threadN]={0.0f};

    for(int i=0; i<matrix_size/blockK; i++)
    {
        float* matAblock = a + (blockDim.y * blockIdx.y) * threadM * matrix_size + i * blockK;
        float* matBblock = b + i * blockK * matrix_size + (blockDim.x * blockIdx.x) * threadN;

        for(int m=0; m<BLOCK_SIZE_Y; m+=threadM)
        {
            for(int n=0; n<blockK; n+=blockDim.x)
            {
                matA[threadIdx.y+m][threadIdx.x+n] = *(matAblock + (threadIdx.y+m) * matrix_size + threadIdx.x+n);
                matB[threadIdx.y+n][threadIdx.x+m] = *(matBblock + (threadIdx.y+n) * matrix_size + threadIdx.x + m);
            }
        }
        __syncthreads();

        for(int j=0; j<blockK; j++)
        {
            for(int k=0; k<threadM; k++)
            {
                threaA[k] = matA[threadM * threadIdx.y + k][j];
            }
            for(int k=0; k<threadN; k++)
            {
                threaB[k] = matB[j][threadN * threadIdx.x + k];
            }
            for(int m=0; m<threadM; m++)
            {
                for(int n=0; n<threadN; n++)
                {
                    threaR[m][n] += threaA[m]*threaB[n];
                }
            }
        }
        __syncthreads();
    }
    
    for(int m=0; m<threadM; m++)
    {
        for(int n=0; n<threadN; n++)
        {
            r[(y+m)*matrix_size+x+n] = threaR[m][n];
        }
    }
}