/****************************************************************************
 *
 * nn_cuda.cu - Neural Network evaluation exploiting CUDA
 *
 * Last updated in 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * To the extent possible under law, the author(s) have dedicated all 
 * copyright and related and neighboring rights to this software to the 
 * public domain worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication
 * along with this software. If not, see 
 * <http://creativecommons.org/publicdomain/zero/1.0/>. 
 *
 * --------------------------------------------------------------------------
 *
 * Compile with:
 * nvcc nn_cuda.cu -o nn_cuda
 *
 * Run with:
 * ./nn_cuda
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

#define BLKDIM 1024
#define R 3

// Print inputs, weights and bias for given layer to screen
void print_layer(int k, int N, double* x, double* W, double* b){
    // Get indices to retrieve elements from arrays
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

// Write inputs, weights and bias for given layer to file
void write_layer(int k, int N, double* x, double* W, double* b, const char* filename){
    FILE* fptr = fopen(filename, "w");
    // Get indices to retrieve elements from arrays
    int x_start=0, x_stop=N, W_start=0, W_stop=(N-R+1)*R;
    for(int i=1; i<k+1; ++i){
        x_start = x_stop;
        W_start = W_stop;
        x_stop += (N - (i)*(R-1));
        W_stop += (N - (i+1)*(R-1))*R;
    }
    fprintf(fptr, "Inputs\n\n");
    for(int i=x_start; i<x_stop; ++i){
        fprintf(fptr, "%lf\t", x[i]);
    }
    fprintf(fptr, "\n\n");
    fprintf(fptr, "Weights\n\n");
    for(int i=W_start; i<W_stop; ++i){
        fprintf(fptr, "%lf\t", W[i]);
    }
    fprintf(fptr, "\n\n");
    fprintf(fptr, "Bias\n\n");
    fprintf(fptr, "%lf\n", b[k]);
    fclose(fptr);
}

// Print array to screen
void print_array(double* a, int n){

    printf("Values\n\n");
    for(int i=0; i<n; ++i){
        printf("%lf\t", a[i]);
    }
    printf("\n");
}

// Write array to file
void write_array(double* a, int n, const char* filename){
    FILE* fptr = fopen(filename, "w");

    fprintf(fptr, "Values\n\n");
    for(int i=0; i<n; ++i){
        fprintf(fptr, "%lf\t", a[i]);
    }
    fclose(fptr);
}


// Fill given array with random values in the range [0,1]
void fill_array(double* array, int n){

	for(int i=0; i<n; ++i)
		array[i] = (double) rand() / RAND_MAX;
}

// Fill given array with random values in the range [0,1]
// until stop index, then fill with 0s.
void partial_fill_array(double* array, int n, int stop){

    for(int i=0; i<stop; ++i)
        array[i] = (double) rand() / RAND_MAX;
    for(int i=stop; i<n; ++i)
        array[i] = (double)0;
}

// Allocate memory to store input and weight arrays
void generate_network(int K, int N, double* h_x, double* h_W, double* h_b, double* d_x, double* d_W, double* d_b, double* d_y){

    // Host
    // Set arrays sizes
    int x_size=N, W_size=(N-R+1)*R, b_size=K;
    for(int k=1; k<K; ++k){
        x_size += (N - (k)*(R-1));
        W_size += (N - (k+1)*(R-1))*R;
    }

    // Memory allocation
	h_x = (double*)malloc(x_size * sizeof(double)); // Input
    h_W = (double*)malloc(W_size * sizeof(double)); // Weights
    h_b = (double*)malloc(b_size * sizeof(double)); // Bias
    
    // Fill with random numbers
    srand(42); // to allow replicability
    // Only first N input values must be filled (others set to 0)
    partial_fill_array(h_x, x_size, N); // Input
    fill_array(h_W, W_size); // Weights
    fill_array(h_b, b_size); // Bias
    
    // Device
    // Memory allocation
    cudaSafeCall(cudaMalloc((void **)&d_x, x_size * sizeof(double))); // Input
    cudaSafeCall(cudaMalloc((void **)&d_W, W_size * sizeof(double))); // Weights
    cudaSafeCall(cudaMalloc((void **)&d_b, b_size * sizeof(double))); // Bias
    
    // Copy from host
    cudaSafeCall(cudaMemcpy(d_x, h_x, x_size * sizeof(double), cudaMemcpyHostToDevice)); // Input
    cudaSafeCall(cudaMemcpy(d_W, h_W, W_size * sizeof(double), cudaMemcpyHostToDevice)); // Weights
    cudaSafeCall(cudaMemcpy(d_b, h_b, b_size * sizeof(double), cudaMemcpyHostToDevice)); // Bias

}

// Define activation function (sigmoid)
__device__ void activation(double* x){
	*x = 1/(1 + exp(-*x));
}

/*
__global__ void test(double* x){
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    x[index] = (double)index;
}
*/

__global__ void kernel(int* indices, double* d_x, double* d_W, double* d_b, double* d_y){
// Kernel function: compute the activations given inputs, weights and bias
    int id = threadIdx.x;
    printf("I am thread #%d: %d,%d,%d,%d\n",id,indices[0],indices[1],indices[2],indices[3]);
    // Matrix multiplication
    for(int i=0; i < indices[1]-R+1; ++i){ // Loop over output neurons
        d_y[i] = d_b[indices[0]]; // Initialize to bias
        for(int j=0; j < R; ++j){
            d_y[i] += (d_x[indices[2]+i+j] * d_W[indices[3]+i*R + j]); // MAC
        }
        activation(&(d_y[i]));
    }
}

// Define propagation function
void forward(int N, int K, double* d_x, double* d_W, double* d_b, double* d_y){
//  Compute activations, applying the kernel function
//  to inputs, weights and biases of each layer, thus obtaining
//  the activations which serve as input for the next one.

    // Compute indices to retrieve each layer's input and weights
    int x_start[K], x_stop[K], W_start[K], W_stop[K];
    x_start[0]=0; x_stop[0]=N; W_start[0]=0; W_stop[0]=(N-R+1)*R;
    for(int i=1; i<K+1; ++i){
        x_start[i] = x_stop[i-1];
        W_start[i] = W_stop[i-1];
        x_stop[i] = x_stop[i-1] + (N - (i)*(R-1));
        W_stop[i] = W_stop[i-1] + (N - (i+1)*(R-1))*R;
    }

    // Loop over layers (except last one)
    for(int k=0; k < K-1; ++k){
        int idxs[4] = {k, N, x_start[k], W_start[k]};
        printf("It. #%d:%d,%d,%d\n", idxs[0], idxs[1], idxs[2], idxs[3]);

        // Compute activations and store them as input for next layer
        //kernel<<<(N-R+1+BLKDIM-1)/BLKDIM, BLKDIM>>>(idxs, d_x, d_W, d_b, d_y);
        kernel<<<1,1>>>(idxs, d_x, d_W, d_b, d_y);
        cudaCheckError();
        cudaSafeCall(cudaMemcpy(d_x+N, d_y, (N-R+1)*sizeof(double), cudaMemcpyDeviceToDevice));
        N = N-R+1;
        
    }
    // Store last activations as output
    //kernel2<<<(N+BLKDIM-1)/BLKDIM, BLKDIM>>>(ls[K-1], output);
    //cudaCheckError();
}




// Define activation function (sigmoid)
void activationh(double* x){
    *x = 1/(1 + exp(-*x));
}


void kernelh(int* indices, double* d_x, double* d_W, double* d_b, double* d_y){
// Kernel function: compute the activations given inputs, weights and bias
    //int id = threadIdx.x;
//    printf("k\tN\tx_start\tW_start:\n%d\t%d\t%d\t%d\n",indices[0],indices[1],indices[2],indices[3]);
    // Matrix multiplication
    int x_idx, W_idx;
    for(int i=0; i < indices[1]-R+1; ++i){ // Loop over output neurons
        d_y[i] = d_b[indices[0]]; // Initialize to bias
        for(int j=0; j < R; ++j){
            x_idx=indices[2]+i+j;
            W_idx=indices[3]+i*R + j;
//            printf("x_idx #%d, W_idx #%d\n", x_idx, W_idx);
            d_y[i] += (d_x[x_idx] * d_W[W_idx]); // MAC
        }
        activationh(&(d_y[i]));
    }
}

// Define propagation function
void forwardh(int N, int K, double* d_x, double* d_W, double* d_b, double* d_y){
//  Compute activations, applying the kernel function
//  to inputs, weights and biases of each layer, thus obtaining
//  the activations which serve as input for the next one.

    // Compute indices to retrieve each layer's input and weights
    int x_start[K], x_stop[K], W_start[K], W_stop[K];
    x_start[0]=0; x_stop[0]=N; W_start[0]=0; W_stop[0]=(N-R+1)*R;
    for(int i=1; i<K+1; ++i){
        x_start[i] = x_stop[i-1];
        W_start[i] = W_stop[i-1];
        x_stop[i] = x_stop[i-1] + (N - (i)*(R-1));
        W_stop[i] = W_stop[i-1] + (N - (i+1)*(R-1))*R;
    }

    // Loop over layers
    for(int k=0; k < K; ++k){
        int idxs[4] = {k, N, x_start[k], W_start[k]};
//        printf("It. #%d:%d,%d,%d\n", idxs[0], idxs[1], idxs[2], idxs[3]);

        // Compute activations and store them as input for next layer
        //kernel<<<(N-R+1+BLKDIM-1)/BLKDIM, BLKDIM>>>(idxs, d_x, d_W, d_b, d_y);
//        for(int i=0; i<24; ++i)
//            printf("%d\t%lf\n", i, d_x[i]);
        kernelh(idxs, d_x, d_W, d_b, d_y);
        memcpy(d_x+x_stop[k], d_y, (N-R+1)*sizeof(double));
        //cudaCheckError();
        //cudaSafeCall(cudaMemcpy(d_x+N, d_y, (N-R+1)*sizeof(double), cudaMemcpyDeviceToDevice));
        N = N-R+1;
        
    }
    // Store last activations as output
    //kernel2<<<(N+BLKDIM-1)/BLKDIM, BLKDIM>>>(ls[K-1], output);
    //cudaCheckError();
}




int main(int argc, char* argv[])
{
    
    int N = 500000; // size of the first layer
    int K = 100; // number of layers
    double tstart, tstop; // timing variables
    double* h_x, *h_W, *h_b; // host copy
    double* d_x, *d_W, *d_b; // device copy
    double* h_y, *d_y; // output

    if(argc>1) {    
        N = atoi(argv[1]);
        K = atoi(argv[2]);
    }

	// Prepare the network: allocate memory, fill weights and bias, fill first layer's input    
    // GENERATE NETWORK

    // Set arrays sizes
    int x_size=N, W_size=(N-R+1)*R, b_size=K, y_size=N-R+1;
    for(int k=1; k<K; ++k){
        x_size += (N - (k)*(R-1));
        W_size += (N - (k+1)*(R-1))*R;
    }

    // Host
    // Memory allocation
    h_x = (double*)malloc(x_size * sizeof(double)); // Input
    h_W = (double*)malloc(W_size * sizeof(double)); // Weights
    h_b = (double*)malloc(b_size * sizeof(double)); // Bias
    h_y = (double*)malloc(y_size * sizeof(double)); // Output

    // Fill with random numbers
    srand(42); // to allow replicability
    // Only first N input values must be filled (others set to 0)
    partial_fill_array(h_x, x_size, N); // Input
    fill_array(h_W, W_size); // Weights
    fill_array(h_b, b_size); // Bias

    // Device
    // Memory allocation
    cudaSafeCall(cudaMalloc((void **)&d_x, x_size * sizeof(double))); // Input
    cudaSafeCall(cudaMalloc((void **)&d_W, W_size * sizeof(double))); // Weights
    cudaSafeCall(cudaMalloc((void **)&d_b, b_size * sizeof(double))); // Bias
    cudaSafeCall(cudaMalloc((void **)&d_y, y_size * sizeof(double))); // Output

    // Copy from host
    cudaSafeCall(cudaMemcpy(d_x, h_x, x_size * sizeof(double), cudaMemcpyHostToDevice)); // Input
    cudaSafeCall(cudaMemcpy(d_W, h_W, W_size * sizeof(double), cudaMemcpyHostToDevice)); // Weights
    cudaSafeCall(cudaMemcpy(d_b, h_b, b_size * sizeof(double), cudaMemcpyHostToDevice)); // Bias

    //cudaMemcpy(h_x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_W, d_W, (N-R+1)*R*sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_b, d_b, 1*sizeof(double), cudaMemcpyDeviceToHost);

//    print_layer(0, N, h_x, h_W, h_b);
    double y_before = h_x[N+1];
	// Set the start time
    tstart = hpc_gettime();
    printf("Before\n");
	// Forward pass: compute activations for each layer, storing the result as input for the next one
    forwardh(N, K, h_x, h_W, h_b, h_y);
//    forward(N, K, d_x, d_W, d_b, d_y);
    // Sinchronize
//    cudaDeviceSynchronize();
    printf("After\n");
    // Set the stop time
    tstop = hpc_gettime();
    // Copy result to host
//    cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost);
    // Print execution time
    printf("Execution time %.2f\n", tstop - tstart);

    //for(int k=0; k<K; ++k) print_layer(k, N, h_x, h_W, h_b);


    double y_true=h_b[0], y_pred=h_x[N+1];
    for(int i=0; i<R; ++i)
        y_true += h_x[i] * h_W[i];
    y_true = 1/(1 + exp(-y_true));


    printf("%lf\t%lf\t%lf\n", y_pred, y_true, y_before);

    //write_array(output, N- K*R + K, "cuda.txt");





/*  TEST
    double* out = (double*)malloc(N*sizeof(double));
    double* d_out;
    cudaMalloc((void **)&d_out,N*sizeof(double));
    kernel1<<<(N+BLKDIM-1)/BLKDIM, BLKDIM>>>(d_out);
    // Sinchronize
    cudaDeviceSynchronize();
    // Set the stop time
    tstop = hpc_gettime();
    // Copy result to host
    cudaMemcpy(out, d_out, N*sizeof(double), cudaMemcpyDeviceToHost);
    // Print execution time
    printf("Execution time %.2f\n", tstop - tstart);
    write_array(out, N, "cuda.txt");
*/

    return EXIT_SUCCESS;
}
