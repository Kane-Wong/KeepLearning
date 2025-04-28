#include <stdio.h>
#include <cuda_runtime.h>
#include<iostream>
#include<sys/time.h>

__global__ void hello_gpu() {
    printf("Hello from GPU\n");
}

__global__ void hello_gpu_thread() {
    printf("Hello from GPU from (block: %d, thread: %d )\n", blockIdx.x, threadIdx.x);
}

__global__ void hello_gpu_thread_2d() {
    printf("Hello from GPU from (block: (%d %d), thread: (%d %d) )\n", 
    blockIdx.x, blockIdx.y, threadIdx.x,  threadIdx.y);
}

int main() {
    dim3 dimGrid(2, 2);
    dim3 dimBlock(8, 16);
    // hello_gpu_thread<<<2, 64>>>();  
    
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    hello_gpu_thread_2d<<<dimGrid, dimBlock>>>();  
    cudaError_t err = cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    float time_use;
    time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);//微秒
    std::cout <<"gpu time cost is "<<time_use/1000/1000<< " ms"<< std::endl;
    
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    return 0;
}
