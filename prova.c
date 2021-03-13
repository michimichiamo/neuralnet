#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define R 3

/*
typedef struct {
   int width;
   int height;
   float* elements;
} Matrix;

*/

// Print inputs, weights and bias for given layer to screen
void print_layer(int k, int N, double* x, double*W, double* b){
    // Get start and stop indices to retrieve elements from arrays
    int x_start=0, x_stop=N, W_start=0, W_stop=(N-R+1)*R;
    for(int i=1; i<k+1; ++i){
        x_start = x_stop;
        W_start = W_stop;
        x_stop += (N - (i)*(R-1));
        W_stop += (N - (i+1)*(R-1))*R;
      }
    
    printf("inputs:\n");
    for(int i=x_start; i<x_stop; ++i){
        printf("%lf\t", x[i]);
    }
    printf("\n");
    printf("weights:\n");
    for(int i=W_start; i<W_stop; ++i){
        printf("%lf\t", W[i]);
    }
    printf("\n");
    printf("bias:\n");
    printf("%lf\n", b[k]);

}

void compute_size(int K, int N){
    int x_stop=N, W_stop=(N-R+1)*R;
    for(int k=1; k<K+1; ++k){
        x_stop += (N - (k)*(R-1));
        W_stop += (N - (k+1)*(R-1))*R;
    }
    printf("%d, %d\n", x_stop, W_stop);
}

// Fill given array with random values in the range [0,1]
void fill_array(double* array, int n){

    for(int i=0; i<n; ++i)
        array[i] = (double) rand() / RAND_MAX;
}

__global__ void kernel(float* d_data){
    int id = threadIdx.x;
    //for(int i=0; i<2; ++i)
    d_data[id] = (float)id;
    }

int main(int argc, char* argv[]){

double* h_data, *d_data;
h_data = (double*) malloc(3*sizeof(double));
cudaSafeCall(cudaMalloc((void **)&d_data, 5*sizeof(double)));

kernel<<<1,3>>>(d_data);

cudaSafeCall(cudaMemcpy(h_data, d_data, 3*sizeof(double), cudaMemcpyDeviceToHost));

for(int i=0; i<3; ++i){
    printf("%f\t", h_data[i]);
}
printf("\n");

/*
    int K = atoi(argv[1]);
    int N = atoi(argv[2]);
    int k = atoi(argv[3]);



    int x_start[K], x_stop[K], W_start[K], W_stop[K];
    x_start[0]=0; x_stop[0]=N; W_start[0]=0; W_stop[0]=(N-R+1)*R;
    for(int i=1; i<K+1; ++i){
        x_start[i] = x_stop[i-1];
        W_start[i] = W_stop[i-1];
        x_stop[i] = x_stop[i-1] + (N - (i)*(R-1));
        W_stop[i] = W_stop[i-1] + (N - (i+1)*(R-1))*R;
    }
    int indices[4] = {x_start[k], x_stop[k], W_start[k], W_stop[k]};


    printf("%d\t%d\t%d\t%d\n", x_start[k], x_stop[k], W_start[k], W_stop[k]);
    printf("%d\t%d\t%d\t%d\n", indices[0], indices[1], indices[2], indices[3]);

    int W_size[K];
    for(int k=0; k<K-1; ++k){
        W_size[k] = W_start[k+1]-W_start[k];
    }
    for(int k=0; k<K-1; ++k){
        printf("%d\t", W_size[k]);
    }
    printf("\n");
*/
    //double x[24], W[54], b[3];
    //fill_array(x, 24);
    //fill_array(W, 54);
    //fill_array(b, 3);
  

    //print_layer(K, 10, x, W, b);
    //compute_size(K, N);

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