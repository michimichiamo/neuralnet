#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
   int width;
   int height;
   float* elements;
} Matrix;

__global__ void kernel(Matrix* mat){
  int id = threadIdx.x;
  mat->elements[id]++;
}

void allocate(Matrix* hs[2], Matrix* ds[2], int size, float* d_elements[2]){
  for(int k=0; k<2; ++k){  
    hs[k] = NULL;
    hs[k] = (Matrix*)malloc(sizeof(Matrix));
    hs[k]->height = 10;
    hs[k]->width = 20;
    hs[k]->elements = (float*)malloc(hs[k]->height * hs[k]->width*sizeof(float)); // initialize it all to '0'

    for(int i=0; i<200; ++i){
      hs[k]->elements[i] = (float)i;
    }

    ds[k] = NULL;
    d_elements[k] = (float*)malloc(200*sizeof(float));

    // allocate the deviceMatrix and d_elements
    cudaSafeCall(cudaMalloc(&ds[k], sizeof(Matrix)));
    cudaSafeCall(cudaMalloc((void **)&(d_elements[k]), size));

    // copy each piece of data separately                                        
    cudaSafeCall(cudaMemcpy(ds[k], hs[k], sizeof(Matrix), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_elements[k], hs[k]->elements, size, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(&(ds[k]->elements), &(d_elements[k]), sizeof(float*), cudaMemcpyHostToDevice));
  }  
}


int main(){

  Matrix* hs[2];
  Matrix* ds[2];
  float* d_elements[2];
  int size = 200 * sizeof(float);

  allocate(hs, ds, size, d_elements);  

  kernel<<<1,200>>>(ds[0]);

  float* h_elements = (float*)malloc(200*sizeof(float));
  cudaMemcpy(h_elements, d_elements[0], size, cudaMemcpyDeviceToHost);
  

  for(int i=0; i<3; ++i){
    printf("%f\t", hs[0]->elements[i]);
  }
  printf("\n");
  for(int i=0; i<3; ++i){
    printf("%f\t", h_elements[i]);
  }
  printf("\n");



}