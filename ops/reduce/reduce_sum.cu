#include "common.cuh"

void _cpu_compute(float *input, float *out, int token_num)
{
    float *anc;
    for (int i = 0; i < token_num; i++)
    {
        anc = input + i * token_num;
        // std::cout << *anc << ' ' << i << std::endl;
        for (int j = 0; j < token_num; j++)
        {
            out[i] += anc[j];
        }
    }
}

__global__ void reduce_kernel_fun0(float *input, float *output)
{
    int tid = threadIdx.x;
    float *x = input + blockDim.x * blockIdx.x;
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            // if(0 == blockIdx.x){
            //     printf("tid: %d, tid_s: %d, a: %f, b: %f \n", tid, tid+s, x[tid], x[tid+s]);
            // }
            x[tid] += x[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
        output[blockIdx.x] = x[0];
}

template <int BLOCK_SIZE>
__global__ void reduce_kernel_fun1(float *input, float *output)
{
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

template <int BLOCK_SIZE>
__global__ void reduce_kernel_fun2(float *input, float *output)
{
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid * 2 * s < blockDim.x)
        {
            int index = tid * 2 * s; 
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

template <int BLOCK_SIZE>
__global__ void reduce_kernel_fun3(float *input, float *output)
{
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

// 计算与访存并行， BLOCK_SIZE 为elements numbers的一半
template <int BLOCK_SIZE>
__global__ void reduce_kernel_fun4(float *input, float *output)
{
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    sdata[tid] = input[i] + input[i + blockDim.x];      
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

template <int BLOCK_SIZE>
__global__ void reduce_kernel_fun5(float *input, float *output)
{
    volatile __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    sdata[tid] = input[i] + input[i + blockDim.x];      
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 32; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32)
    {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

__global__ void reduce_kernel_fun6(float *input, float *output, const int M, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    for(int i = bid; i < M; i += gridDim.x){
        float* row_pointer = input + i * N;
        float sum_val = 0;
        for(int j = tid; j < N; j += warpSize)
        {
            sum_val += row_pointer[j];
        }
        for(int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            sum_val += __shfl_xor_sync(0xffffffff, sum_val, offset);
        }
        output[i] = sum_val;
    }
}

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