#include "common.h"

void printDeviceInfor(int deviceID)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceID);
    std::cout << "GPU device " << deviceID << ": " << prop.name << std::endl;
}