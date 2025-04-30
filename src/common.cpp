#include "common.h"

timerecord::timerecord()
{
    gettimeofday(&realtime, NULL); 
}

void timerecord::freshtime()
{
    gettimeofday(&realtime, NULL); 
}

void printTimeGap(timerecord start, timerecord end, std::string prefix)
{
    float cost = (end.realtime.tv_sec-start.realtime.tv_sec)*1000000+(end.realtime.tv_usec-start.realtime.tv_usec);//微秒
    std::cout << prefix << " time cost is " << cost / 1000 << " ms" << std::endl;
}