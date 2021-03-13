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
#define R 5

// Define struct to store inputs, weights and bias for each layer
typedef struct {
	int* N;
	double* x;
	double* W;
	double* b;
} layer_t;

// Print inputs, weights and bias for given layer to screen
void print_layer(layer_t l){
	printf("inputs:\n");
	for(int i=0; i<l.N[0]; ++i){
		printf("%lf\t", l.x[i]);
	}
	printf("\n");
	printf("weights:\n");
	for(int i=0; i<l.N[0] * R; ++i){
		printf("%lf\t", l.W[i]);
	}
	printf("\n");
	printf("bias:\n");
	printf("%lf\n", l.b[0]);
}

// Write inputs, weights and bias for given layer to file
void write_layer(layer_t l, const char* filename){
    FILE* fptr = fopen(filename, "w");

    fprintf(fptr, "Inputs\n\n");
    for(int i=0; i<l.N[0]; ++i){
        fprintf(fptr, "%lf\t", l.x[i]);
    }
    fprintf(fptr, "\n\n");
    fprintf(fptr, "Weights\n\n");
    for(int i=0; i<l.N[0] * R; ++i){
        fprintf(fptr, "%lf\t", l.W[i]);
    }
    fprintf(fptr, "\n\n");
    fprintf(fptr, "Bias\n\n");
    fprintf(fptr, "%lf\n", l.b[0]);
    fclose(fptr);
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

// Allocate memory to store input and weight arrays of each layer
void initialize_network(layer_t* ls, layer_t* d_ls, int N, int K){

//	printf("Allocating memory...\n");
	for(int k=0; k<K; ++k){ 
        // Host
        // Allocate memory for # of input neurons
        ls[k].N = (int*)malloc(sizeof(int));
		// Set # of input neurons
    	ls[k].N[0] = N - k * (R-1);
    	// Allocate memory for input array
		ls[k].x = (double*)malloc(ls[k].N[0] * sizeof(double));
		// Set # of weights (R elements for each input neuron)
    	int w_N = ls[k].N[0] * R;
		// Allocate memory for weights array
  		ls[k].W = (double*)malloc(w_N *sizeof(double));
        // Allocate memory for bias
        ls[k].b = (double*)malloc(sizeof(double));

        // Device
        // Allocate memory for # of input neurons
        cudaSafeCall(cudaMalloc((void **)&(d_ls[k].N), sizeof(int)));
        // Allocate memory for input array
        cudaSafeCall(cudaMalloc((void **)&(d_ls[k].x), ls[k].N[0] * sizeof(double)));
        // Allocate memory for weights array
        cudaSafeCall(cudaMalloc((void **)&(d_ls[k].W), w_N *sizeof(double)));
        // Allocate memory for bias
        cudaSafeCall(cudaMalloc((void **)&(d_ls[k].b), sizeof(double)));

    }

}

void generate_values(layer_t* ls, layer_t* d_ls, int K){

//    printf("Generating input values, weights and biases...\n");
    for(int k=0; k<K; ++k){
        // Host
        // Set # of weights (R elements for each input neuron)
        int w_N = ls[k].N[0] * R;
        // Fill weights
        fill_array(ls[k].W, w_N);
        // Fill bias
        ls[k].b[0] = (double) rand()/RAND_MAX;

        // Device
        // Fill weights
        cudaSafeCall(cudaMemcpy(d_ls[k].N, ls[k].N, sizeof(int), cudaMemcpyHostToDevice));
        // Fill weights
        cudaSafeCall(cudaMemcpy(d_ls[k].W, ls[k].W, w_N*sizeof(double), cudaMemcpyHostToDevice));
        // Fill bias
        cudaSafeCall(cudaMemcpy(d_ls[k].b, ls[k].b, sizeof(double), cudaMemcpyHostToDevice));

    }
    // Fill input array for first layer with random values in the range [0,1]
    fill_array(ls[0].x, ls[0].N[0]);
    cudaSafeCall(cudaMemcpy(d_ls[0].x, ls[0].x, ls[0].N[0]*sizeof(double), cudaMemcpyHostToDevice));

}

void generate_network(layer_t* ls, layer_t* d_ls, int N, int K){
// Allocate memory and populate layers weights and bias with random values in the range [0,1]
//    printf("Generating network...\n");
    // Allocate memory
    initialize_network(ls, d_ls, N, K);
    // Generate values
    generate_values(ls, d_ls, K);

//    printf("Network generated.\n");
}


// Define activation function (sigmoid)
__device__ void activation(double* x){
	*x = 1/(1 + exp(-*x));
}

__global__ void kernel(double* x, double* W, double b, int N, double* y){
// Kernel function: given layer (inputs, weights, bias),
// compute the activations

    //__shared__ double input[N];
    //const int index = threadIdx.x + blockIdx.x * blockDim.x;

    // Matrix multiplication
    for(int i=0; i < N - R + 1; ++i){ // Loop over output neurons
        y[i] = b; // Initialize to bias
        for(int j=0; j < R; ++j){
            y[i] += (x[i + j] * W[i*R + j]); // MAC
        }
        activation(&y[i]);
    }
}

/*
__global__ void test(double* x){
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    x[index] = (double)index;
}
*/

// Define propagation function
void forward(layer_t* h_ls, layer_t* d_ls, int K, double* d_output){
//  Compute activations, applying the kernel function
//  to inputs, weights and biases of each layer, thus obtaining
//  the activations which serve as input for the next one.

    // Loop over layers (except last one)
    for(int k=0; k < K-1; ++k){
printf("Before\n");
        // Compute activations and store them as input for next layer
        kernel<<<(h_ls[k].N[0]+BLKDIM-1)/BLKDIM, BLKDIM>>>(d_ls[k].x, d_ls[k].W, d_ls[k].b[0], d_ls[k].N[0], d_ls[k+1].x);
printf("After\n");
        //printf("Layer #%d\n", k);
        cudaCheckError();
        //cudaDeviceSynchronize();
        // Free useless memory
        //cudaFree(d_ls[k].x);
        //cudaFree(d_ls[k].W);
    }
    // Store last activations as output
    kernel<<<(h_ls[K-1].N[0]+BLKDIM-1)/BLKDIM, BLKDIM>>>(d_ls[K-1].x, d_ls[K-1].W, d_ls[K-1].b[0], d_ls[K-1].N[0], d_output);
}

int main(int argc, char* argv[])
{
    
    int N = 500000; // size of the first layer
    int K = 100; // number of layers
    double tstart, tstop; // timing variables
    double* h_output; // host copy
    double* d_output; // device copy

    if(argc>1) {    
        N = atoi(argv[1]);
        K = atoi(argv[2]);

    }

	// Instantiate a struct for each layer to store inputs, weights and bias
	// (plus one to store the final output)
    layer_t h_ls[K]; // host copy
    layer_t d_ls[K]; // device copy

	// Set the seed
    srand(42);

	// Prepare the network: allocate memory, fill weights and bias, fill first layer's input
    generate_network(h_ls, d_ls, N, K);

    // Allocate memory for last output
    size_t out_size = (N - K*R + K)*sizeof(double);
    h_output = (double*)malloc(out_size);
    cudaSafeCall(cudaMalloc((void **)&d_output, out_size));

	// Set the start time
    tstart = hpc_gettime();
	// Forward pass: compute activations for each layer, storing the result as input for the next one
    forward(h_ls, d_ls, K, d_output);

    //print_layer(d_ls[0]);
    //printf("%lf\n", h_ls[0].x[0]);
    //printf("%lf\n", d_ls[0].x[0]);

/*    
    for(int k=0; k<K; k++){
        print_layer(d_ls[k]);
    }
*/
    // Sinchronize
    cudaDeviceSynchronize();
    // Set the stop time
    tstop = hpc_gettime();
    // Copy result to host
    cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost);
    // Print execution time
    printf("Execution time %.2f\n", tstop - tstart);

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
