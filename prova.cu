#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define R 3

typedef struct {
   int width;
   int height;
   float* elements;
} Matrix;

__global__ void kernel(float* d_data){
    int id = threadIdx.x;
    d_data[id]++;
    printf("I am thread #%d: data:%f\n", id, d_data[id]);
    }

int main(int argc, char* argv[]){

    float* h_data, *h_out, *d_data, *d_out;
    h_data = (float*) malloc(3*sizeof(float));
    h_out = (float*) malloc(3*sizeof(float));
    cudaSafeCall(cudaMalloc((void **)&d_data, 5*sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&d_out, 5*sizeof(float)));
    cudaSafeCall(cudaMemset(d_data, 0, 5*sizeof(float)));
    cudaSafeCall(cudaMemset(d_out, 0, 5*sizeof(float)));
    
    for(int i=0; i<3; ++i){
        kernel<<<1,i+1>>>(d_data);
        cudaSafeCall(cudaMemcpy(d_out+2, d_data, (i+1)*sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    cudaSafeCall(cudaMemcpy(h_data, d_data, 3*sizeof(float), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(h_out, d_out, 5*sizeof(float), cudaMemcpyDeviceToHost));
    
    for(int i=0; i<5; ++i){
        printf("%f\t", h_data[i]);
        printf("%f\n", h_out[i]);
    }
    printf("\n");

/*
    float* d_data; float* h_data;
    int R = 10, C = 15;
    int N = R*C;
    int size = N * sizeof(float);

    h_data = (float*)malloc(size);

    for(int i=0; i<C; ++i){
        for(int j=0; j<R; ++j){
            h_data[i*R + j]= (float)(i*R + j);
            printf("%f\t", h_data[i*R + j]);
      }
    }
    printf("\n");
    // allocate device memory
    cudaSafeCall(cudaMalloc((void **)&(d_data), size));
    // copy host array to device                                    
    cudaSafeCall(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    
    for(int i=0; i<3; ++i){
        printf("%f\t", h_data[i]);
    }
    printf("\n");

    kernel<<<1,N>>>(d_data);
    cudaDeviceSynchronize();

    cudaSafeCall(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    for(int i=0; i<3; ++i){
        printf("%f\t", h_data[i]);
    }
    printf("\n");
*/


}