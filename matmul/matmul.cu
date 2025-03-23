#include<iostream>
#include<sys/time.h>
using namespace std;

void _cpu_compute(float *a, float *b, float *r, int matrix_size){
    for (int i=0; i<matrix_size; i++){
        for (int j=0; j<matrix_size; j++){
            for (int k=0; k<matrix_size; k++){
                r[i*matrix_size+j] += a[i*matrix_size+k] * b[k*matrix_size+j];
            }
        }
    }
}

__global__ void _cuda_compute(float *a, float *b, float *r, int matrix_size){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    // printf("%d, %d, %d, %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
    if (x<matrix_size && y<matrix_size){
        for (int k=0; k<matrix_size; k++){
            r[x*matrix_size+y] += a[x*matrix_size+k] * b[k*matrix_size+y];
        }
    }
}

int main(){
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
    size_t block_x=32, block_y=1;
    size_t grid_x = (matrix_size+block_x -1) / block_x;
    size_t grid_y = (matrix_size+block_y-1) / block_y;
    dim3 dimGrid(grid_y, grid_x);
    dim3 dimBlock(block_y, block_x);

    struct timeval start;
    struct timeval end;
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    gettimeofday(&start, NULL);
    _cuda_compute<<<dimGrid, dimBlock>>>(d_a, d_b, d_r, matrix_size);
    cudaDeviceSynchronize(); // 同步，且检查执行期间发生的错误
    gettimeofday(&end, NULL);
    cudaMemcpy(r_re, d_r, size, cudaMemcpyDeviceToHost);
    float time_use;
    time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);//微秒
    std::cout <<"gpu time cost is "<<time_use/1000<< " ms"<< std::endl;

    gettimeofday(&start, NULL);
    _cpu_compute(a, b, r, matrix_size);
    gettimeofday(&end, NULL);
    time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);//微秒
    std::cout <<"cpu time cost is "<<time_use/1000<< " ms"<< std::endl;


    bool error=false;
    for(int i=0; i<matrix_size*matrix_size; i++){
        float err = fabs(r[i]-r_re[i]);
        if(err>1.0e-10){
            error = true;
        }
    }
    printf("Result: %s\n", error?"Errors":"Passed");

    free(a);
    free(b);
    free(r);
    free(r_re);
    cudaFreeHost(d_a);
    cudaFreeHost(d_b);
    cudaFreeHost(d_r);
   
    return 0;
}