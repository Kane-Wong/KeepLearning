#include "matmul.h"

int main(int argc, char **argv)
{
    assert(argc == 2);
    const int case_number = atoi(argv[1]);
    printf("cpu running case %d\n", case_number);
    
    // 1. 环境准备
    size_t size = matrix_size * matrix_size * sizeof(float);
    float *a = (float *)malloc(size);
    float *b = (float *)malloc(size);
    float *r = (float *)malloc(size);
    for(int i=0; i<matrix_size*matrix_size; i++){
        a[i] = rand() % matrix_size;
        b[i] = rand() % matrix_size;
        r[i] = 0;
    }

    // 2. cpu 计算
    timerecord start, end;
    start.freshtime();
    for(int l=0; l<10; l++)
    {
        switch (case_number)
        {
        case 0:
            _cpu_compute_fun0(a, b, r, matrix_size);
            break;
        case 1:
            _cpu_compute_fun1(a, b, r, matrix_size);
            break;
        default:
            printf("Error: Invalid cpu running case: %d\n", case_number);
            return EXIT_FAILURE;
        }
    }
    end.freshtime();
    printTimeGapLoop(start, end, 10, "cpu");

    // 3. 环境释放
    free(a);
    free(b);
    free(r);
   
    return 0;
}