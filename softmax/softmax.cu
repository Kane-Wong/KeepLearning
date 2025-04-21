#include<iostream>
#include <unordered_map>
#include <functional>
#include<sys/time.h>
#include <cassert>
#define M (1024)
#define N (2048)

void softmax_cpu_fun0(float *input, float *output)
{
    float* row_in = input;
    float* row_out = output;
    for(int i=0; i<M; i++)
    {
        float max_value = *row_in;
        float sum_row = 0;
        for(int j=1; j<N; j++)
        {
            max_value = max(max_value, row_in[j]);
        }
        for(int j=0; j<N; j++)
        {
            row_out[j] = exp(row_in[j] - max_value);
            sum_row += row_out[j];
        }
        for(int j=0; j<N; j++)
        {
            row_out[j] /= sum_row;
        }
        row_in += N;
        row_out += N;
    }
}

void softmax_cpu_fun1(float* input, float* output) {
    for (int m = 0; m < M; ++m) {
        float maxval = -INFINITY;
        const float* x = input + m * N;
        for (int n = 0; n < N; ++n) {
            maxval = maxval > x[n] ? maxval : x[n];
        }
        float sumval = 0.0f;
        for (int n = 0; n < N; ++n) {
            sumval += exp(x[n] - maxval);
        }
        float* y = output + m * N;
        for (int n = 0; n < N; ++n) {
            y[n] = exp(x[n] - maxval) / sumval;
        }
    }
}

void softmax_cpu_fun2(float *input, float *output)
{
    float* row_in = input;
    float* row_out = output;
    for(int i=0; i<M; i++)
    {
        float max_value = *row_in;
        float max_value_pre = max_value;
        float sum_row = 0;
        for(int j=1; j<N; j++)
        {
            max_value_pre = max_value;
            max_value = max(max_value, row_in[j]);
            sum_row = sum_row * exp(max_value_pre - max_value) + exp(row_in[j] - max_value);
        }
        for(int j=0; j<N; j++)
        {
            row_out[j] = exp(row_in[j] - max_value) / sum_row;
        }
        row_in += N;
        row_out += N;
    }
}


// 开启M个线程，对应M行
__global__ void softmax_gpu_fun0(float* input, float* output)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float* row_in = input + tid * N;
    float* row_out = output + tid * N;
    float max_value = *row_in;
    float sum_row = 0;
    if(tid<M)
    {
        for(int j=1; j<N; j++)
        {
            max_value = max(max_value, row_in[j]);
        }
        for(int j=0; j<N; j++)
        {
            sum_row +=  exp(row_in[j] - max_value);
        }
        for(int j=0; j<N; j++)
        {
            row_out[j] = exp(row_in[j] - max_value) / sum_row;
        }
    }
}

// 开启 M*N 个线程，归约求 max 和 sum， 最大支持横轴点数：1024
template<int BLOCK_SIZE>
__global__ void softmax_gpu_fun1(float* input, float* output)
{
    __shared__ float sdata[BLOCK_SIZE];     // block内共享
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];  
    __syncthreads();

    // 求每行最大值, 最终保存在sdata[0]
    // float max_value = -INFINITY;
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    // if (tid == 0) max_value = sdata[0];

    // 求指数
    sdata[tid] = exp(input[i] - sdata[0]);
    output[i] = sdata[tid];
    __syncthreads();

    // 指数求和, 最终保存在sdata[0]
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    output[i] /= sdata[0];
}

// 较fun1 可支持更大的横轴点数，
__global__ void softmax_gpu_fun2(float* input, float* output, const int M_, const int N_) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    assert(blockDim.x == warpSize); 
    for (int i = bid; i < M_; i += gridDim.x) {
        const float* row_pointer_in = input + i * N_;
        float* row_pointer_out = output + i * N_;

        float maxval = -INFINITY;
        for (int j = tid; j < N_; j += warpSize) {
            maxval = max(maxval, row_pointer_in[j]);
        }
        // warp-reduce to calculate the MAX of maxval among all lanes
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            maxval = max(maxval, __shfl_xor_sync(0xFFFFFFFF, maxval, offset));
        }

        float sumval = 0.0f;
        for (int j = tid; j < N_; j += warpSize) {
            sumval += exp(row_pointer_in[j] - maxval);
        }
        // warp-reduce to calculate the SUM of sumval among all lanes
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sumval += __shfl_xor_sync(0xFFFFFFFF, sumval, offset);
        }
        
        for (int j = tid; j < N_; j += warpSize) {
            row_pointer_out[j] = exp(row_pointer_in[j] - maxval) / sumval ;
        }
    }
}

// fun0 online softmax
__global__ void softmax_gpu_fun3(float* input, float* output)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float* row_in = input + tid * N;
    float* row_out = output + tid * N;
    float max_value = *row_in;
    float max_value_pre = max_value;
    float sum_row = 0;
    if(tid<M)
    {
        for(int j=1; j<N; j++)
        {
            max_value_pre = max_value;
            max_value = max(max_value, row_in[j]);
            sum_row = sum_row * exp(max_value_pre - max_value) + exp(row_in[j] - max_value);
        }
        for(int j=0; j<N; j++)
        {
            row_out[j] = exp(row_in[j] - max_value) / sum_row;
        }
    }
}

// fun2 online softmax
__global__ void softmax_gpu_fun4(float* input, float* output, const int M_, const int N_) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    assert(blockDim.x == warpSize); 
    for (int i = bid; i < M_; i += gridDim.x) {
        const float* row_pointer_in = input + i * N_;
        float* row_pointer_out = output + i * N_;

        float maxval = -INFINITY, bigger;
        float sumval = 0.0f;
        for (int j = tid; j < N_; j += warpSize) {
            bigger  = max(maxval, row_pointer_in[j]);
            sumval = sumval * exp(maxval - bigger) + exp(row_pointer_in[j] - bigger); 
            maxval = bigger;
        }
        
        // warp-reduce to calculate the MAX of maxval among all lanes
        // #pragma unroll
        float offsetMax, offsetSum;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            __syncwarp();
            offsetMax = __shfl_xor_sync(0xFFFFFFFF, maxval, offset);
            offsetSum = __shfl_xor_sync(0xFFFFFFFF, sumval , offset);
            if(offsetMax > maxval)
            {
                sumval *= exp(maxval - offsetMax);
                maxval = offsetMax;
            }
            else
            {
                offsetSum *= exp(offsetMax - maxval);
            }
            sumval += offsetSum;
        }
        
        for (int j = tid; j < N_; j += warpSize) {
            row_pointer_out[j] = exp(row_pointer_in[j] - maxval) / sumval ;
        }
    }
}

int main()
{
    int deviceID = 0;
    cudaSetDevice(deviceID);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceID);
    std::cout << "GPU device " << deviceID << ": " << prop.name << std::endl;

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

    struct timeval start, end;
    float time_use;
    gettimeofday(&start, NULL);
    // softmax_gpu_fun0<<<dim3(M), dim3(1)>>>(d_a, d_r);
    // softmax_gpu_fun1<N><<<dim3(M), dim3(N)>>>(d_a, d_r);
    // softmax_gpu_fun2<<<dim3(M), dim3(32)>>>(d_a, d_r, M, N);
    // softmax_gpu_fun3<<<dim3(M), dim3(1)>>>(d_a, d_r);
    softmax_gpu_fun4<<<dim3(M), dim3(32)>>>(d_a, d_r, M, N);
    // online_softmax_kernel3<<<dim3(M), dim3(32)>>>(d_a, d_r, M, N);
    cudaDeviceSynchronize(); 
    cudaMemcpy(output_host_cuda, d_r, data_size, cudaMemcpyDeviceToHost);
    gettimeofday(&end, NULL);
    time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);//微秒
    std::cout <<"gpu time cost is "<<time_use/1000<< " ms"<< std::endl;
    
    gettimeofday(&start, NULL);
    softmax_cpu_fun0(input_host, output_host_cpu);
    gettimeofday(&end, NULL);
    time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);//微秒
    std::cout <<"cpu time cost is "<<time_use/1000<< " ms"<< std::endl;

    bool error=false;
    for(int i=0; i<M*N; i++){
        float err = fabs(output_host_cpu[i]-output_host_cuda[i]);
        // if(i < 10) std::cout << ' '<< output_host_cpu[i] << ' ' << output_host_cuda[i] << ' ' << i << std::endl; 
        if(err>1e-4){
            error = true;
            if(i < 10) std::cout << ' '<< output_host_cpu[i] << ' ' << output_host_cuda[i] << ' ' << i << std::endl; 
        }
    }
    printf("Result: %s\n", error?"Errors":"Passed");
    return 0;
}