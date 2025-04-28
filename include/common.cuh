
#include<iostream>
#include<sys/time.h>
#include <cassert>

struct timerecord{
    struct timeval realtime;
    timerecord(){ gettimeofday(&realtime, NULL); };
    void freshtime(){ gettimeofday(&realtime, NULL); };
};

void printDeviceInfor(int deviceID)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceID);
    std::cout << "GPU device " << deviceID << ": " << prop.name << std::endl;
}

void printTimeGap(timerecord start, timerecord end, std::string prefix)
{
    float cost = (end.realtime.tv_sec-start.realtime.tv_sec)*1000000+(end.realtime.tv_usec-start.realtime.tv_usec);//微秒
    std::cout << prefix << " time cost is " << cost / 1000 << " ms" << std::endl;
}

template<typename T>
void validResult(T* golden_result, T* compare_result, int data_size, float max_diff)
{
    int error_num = 0;
    for(int i=0; i<data_size; i++){
        float err = fabs(golden_result[i]-compare_result[i]);
        // if(i < 10) std::cout << ' '<< golden_result[i] << ' ' << compare_result[i] << ' ' << i << std::endl; 
        if(err>max_diff){
            error_num += 1;
            if(error_num < 10) std::cout << ' '<< golden_result[i] << ' ' << compare_result[i] << ' ' << i << std::endl; 
        }
    }
    printf("Result: %s\n", error_num>0?"Errors":"Passed");
}
