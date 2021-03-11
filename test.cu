#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include "hpc.h"

typedef struct S {
    int *arr1;
    int *arr2;
    int *arr3; 
    int *count;
} S;

const int size = 10000;

__global__ void some_kernel(S *s)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
    {
        s->arr1[id] = 1; // val1
        s->arr2[id] = 2; // val2
        s->arr3[id] = 3; // val3
        atomicAdd(s->count, 1);
    }
}

int main(){
int *host_arr1, *host_arr2, *host_arr3;
int *dev_arr1, *dev_arr2, *dev_arr3;
int *host_count, *dev_count;

// Allocate and fill host data
host_arr1 = (int*)malloc(size*sizeof(int));
host_arr2 = (int*)malloc(size*sizeof(int));
host_arr3 = (int*)malloc(size*sizeof(int));
host_count = (int*)malloc(sizeof(int));
//host_count[0] = 0;

// Allocate device data   
cudaMalloc((void **) &dev_arr1, size * sizeof(*dev_arr1));
cudaMalloc((void **) &dev_arr2, size * sizeof(*dev_arr2));
cudaMalloc((void **) &dev_arr3, size * sizeof(*dev_arr3));
cudaMalloc((void **) &dev_count, sizeof(*dev_count));

// Allocate helper struct on the device
S *dev_s;
cudaMalloc((void **) &dev_s, sizeof(*dev_s));


// Copy data from host to device
cudaMemcpy(dev_arr1, host_arr1, size * sizeof(*dev_arr1), cudaMemcpyHostToDevice);
cudaMemcpy(dev_arr2, host_arr2, size * sizeof(*dev_arr2), cudaMemcpyHostToDevice);
cudaMemcpy(dev_arr3, host_arr3, size * sizeof(*dev_arr3), cudaMemcpyHostToDevice);
cudaMemcpy(dev_count, host_count, sizeof(*dev_count), cudaMemcpyHostToDevice);

// NOTE: Binding pointers with dev_s
cudaMemcpy(&(dev_s->arr1), &dev_arr1, sizeof(dev_s->arr1), cudaMemcpyHostToDevice);
cudaMemcpy(&(dev_s->arr2), &dev_arr2, sizeof(dev_s->arr2), cudaMemcpyHostToDevice);
cudaMemcpy(&(dev_s->arr3), &dev_arr3, sizeof(dev_s->arr3), cudaMemcpyHostToDevice);
cudaMemcpy(&(dev_s->count), &dev_count, sizeof(dev_s->count), cudaMemcpyHostToDevice);



// Call kernel
some_kernel<<<10000/256 + 1, 256>>>(dev_s); // block size need to be a multiply of 256

// Copy result to host:
cudaMemcpy(host_arr1, dev_arr1, size * sizeof(*host_arr1), cudaMemcpyDeviceToHost);
cudaMemcpy(host_arr2, dev_arr2, size * sizeof(*host_arr2), cudaMemcpyDeviceToHost);
cudaMemcpy(host_arr3, dev_arr3, size * sizeof(*host_arr3), cudaMemcpyDeviceToHost);
cudaMemcpy(host_count, dev_count, sizeof(*host_count), cudaMemcpyDeviceToHost);

// Print some result
printf("%d\n", host_arr1[size-1]);
printf("%d\n", host_arr2[size-1]);
printf("%d\n", host_arr3[size-1]);
printf("%d\n", host_count[0]);

}
