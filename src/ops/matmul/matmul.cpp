#include "matmul.h"

void _cpu_compute_fun0(float *a, float *b, float *r, int matrix_size){
    for (int i=0; i<matrix_size; i++){
        for (int j=0; j<matrix_size; j++){
            for (int k=0; k<matrix_size; k++){
                r[i*matrix_size+j] += a[i*matrix_size+k] * b[k*matrix_size+j];
            }
        }
    }
}

void _cpu_compute_fun1(float *a, float *b, float *r, int matrix_size){
    for (int i=0; i<matrix_size; i++){
        for (int j=0; j<matrix_size; j++){
            float sum=0.0;
            for (int k=0; k<matrix_size; k++){
                sum += a[i*matrix_size+k] * b[k*matrix_size+j];
            }
            r[i*matrix_size+j] = sum;
        }
    }
}