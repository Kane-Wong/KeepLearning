#include "common.cuh"
using namespace std;

void _cpu_compute_fun0(float *a, float *b, float *r, int matrix_size){
    for (int i=0; i<matrix_size; i++){
        for (int j=0; j<matrix_size; j++){
            for (int k=0; k<matrix_size; k++){
                r[i*matrix_size+j] += a[i*matrix_size+k] * b[k*matrix_size+j];
            }
        }
    }
}

void _cpu_compute_fun1(float *a, float *b, float *r, int matrix_size){
    for (int i=0; i<matrix_size; i++){
        for (int j=0; j<matrix_size; j++){
            float sum=0.0;
            for (int k=0; k<matrix_size; k++){
                sum += a[i*matrix_size+k] * b[k*matrix_size+j];
            }
            r[i*matrix_size+j] = sum;
        }
    }
}

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

int main(int argc, char **argv)
{
    int cpu_case = atoi(argv[1]);
    int gpu_case = atoi(argv[2]);
    printf("cpu running case %d, gpu running case %d\n", cpu_case, gpu_case);
    
    int deviceID = 0;
    cudaSetDevice(deviceID);
    printDeviceInfor(deviceID);
    
    // 1. 环境准备
    const int matrix_size = 1024;
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
    dim3 dimGrid(grid_y, grid_x);
    dim3 dimBlock(block_y, block_x);
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
        _cuda_compute_fun3<block_x, block_y><<<dimGrid, dim3(block_y/8, block_x/8)>>>(d_a, d_b, d_r, matrix_size); 
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