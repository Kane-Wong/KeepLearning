
#include<iostream>
#include<sys/time.h>
#include <cassert>
#include <cmath>
#pragma once

struct timerecord{
    struct timeval realtime;
    timerecord();
    void freshtime();
};

void printDeviceInfor(int deviceID);

void printTimeGap(timerecord start, timerecord end, std::string prefix);
void printTimeGapLoop(timerecord start, timerecord end, int loop_num, std::string prefix);

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