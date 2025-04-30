#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
int main()
{
  int count = 0;
  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;
  if (count == 0) return -1;
  for (int device = 0; device < count; ++device)
  {
    cudaDeviceProp prop;
    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))
    std::cout << "GPU device " << device << ": " << prop.name << std::endl;
    std::cout << "SM的数量：" << prop.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << prop.maxThreadsPerBlock << std::endl;
    std::cout << "每个SM的最大线程数：" << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个SM的最大线程束数：" << prop.maxThreadsPerMultiProcessor / 32 << std::endl;
  }
  return 0;
}
