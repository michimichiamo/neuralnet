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
	int N;
	double* x;
	double* W;
	double b;
} layer_t;

// Print inputs, weights and bias for given layer to screen
void print_layer(layer_t* l){
	printf("inputs:\n");
	for(int i=0; i<l->N; ++i){
		printf("%lf\t", l->x[i]);
	}
	printf("\n");
	printf("weights:\n");
	for(int i=0; i<l->N * R; ++i){
		printf("%lf\t", l->W[i]);
	}
	printf("\n");
	printf("bias:\n");
	printf("%lf\n", l->b);
}

// Write inputs, weights and bias for given layer to file
void write_layer(layer_t* l, const char* filename){
    FILE* fptr = fopen(filename, "w");

    fprintf(fptr, "Inputs\n\n");
    for(int i=0; i<l->N; ++i){
        fprintf(fptr, "%lf\t", l->x[i]);
    }
    fprintf(fptr, "\n\n");
    fprintf(fptr, "Weights\n\n");
    for(int i=0; i<l->N * R; ++i){
        fprintf(fptr, "%lf\t", l->W[i]);
    }
    fprintf(fptr, "\n\n");
    fprintf(fptr, "Bias\n\n");
    fprintf(fptr, "%lf\n", l->b);
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
void generate_network(int K, int N, layer_t** h_ls, layer_t** d_ls){
//	printf("Allocating memory...\n");

    // Allocate memory for input, fill weights and bias
	for(int k=0; k<K; ++k){
        
        // Host
        // Allocate memory for struct
        h_ls[k] = NULL;
        h_ls[k] = (layer_t*)malloc(sizeof(layer_t));
        // Memory allocation
        //h_ls[k]->N = *(int*)malloc(sizeof(int));
        h_ls[k]->N = N - k * (R-1); // Set # of input neurons
		h_ls[k]->x = (double*)malloc(h_ls[k]->N * sizeof(double)); // Input array
  		int w_N = h_ls[k]->N * R; // Set # of weights (R elements for each input neuron)
        h_ls[k]->W = (double*)malloc(w_N *sizeof(double)); // Weights array
        //h_ls[k]->b = *(double*)malloc(sizeof(double));
        // Fill
        fill_array(h_ls[k]->W, w_N); // Weights array
        h_ls[k]->b = (double) rand()/RAND_MAX; // Bias
        
        // Device
        double* d_x;//, *d_b; int* d_N; // device data
        double* d_W;
        // Memory allocation
        d_ls[k] = NULL;
        cudaSafeCall(cudaMalloc(&(d_ls[k]), sizeof(layer_t))); // struct
        //cudaSafeCall(cudaMalloc((void **)&d_N, sizeof(int))); // # of input neurons
        cudaSafeCall(cudaMalloc((void **)&d_x, h_ls[k]->N * sizeof(double))); // Input array
        cudaSafeCall(cudaMalloc((void **)&d_W, w_N *sizeof(double))); // Weights array
        //cudaSafeCall(cudaMalloc((void **)&d_b, sizeof(double))); // Bias
        
        // Copy from host
        cudaSafeCall(cudaMemcpy(d_ls[k], h_ls[k], sizeof(layer_t*), cudaMemcpyHostToDevice)); // # struct
        //cudaSafeCall(cudaMemset(d_N, h_ls[k]->N, sizeof(int))); // # of input neurons
        cudaSafeCall(cudaMemcpy(d_W, h_ls[k]->W, w_N*sizeof(double), cudaMemcpyHostToDevice)); // Weights
        //cudaSafeCall(cudaMemset(d_b, h_ls[k]->b, sizeof(double))); // Bias
        printf("Before1\n");
        // Link (point to device arrays from structure)
        //cudaSafeCall(cudaMemset(d_ls[k]->N, h_ls[k]->N, sizeof(int)));
        cudaSafeCall(cudaMemcpy(&(d_ls[k]->W), &d_W, w_N*sizeof(double), cudaMemcpyHostToDevice));
        //cudaSafeCall(cudaMemset(d_ls[k]->b, h_ls[k]->b, sizeof(double)));
        printf("After1\n");
    }
    // Fill first input
    // Memory allocation
    double* d_x0; // device copy
    cudaSafeCall(cudaMalloc((void **)&d_x0, N * sizeof(double))); // memory allocation
    // Fill
    fill_array(h_ls[0]->x, N); // Fill with random values in the range [0,1]
    cudaSafeCall(cudaMemcpy(d_x0, h_ls[0]->x, N*sizeof(double), cudaMemcpyHostToDevice)); // copy to device
    // Link (point to device arrays from structure)
    cudaSafeCall(cudaMemcpy(d_ls[0]->x, d_x0, N*sizeof(double), cudaMemcpyDeviceToDevice));
}

/*
// Allocate memory to store input and weight arrays of each layer
void initialize_network(int N, int K, layer_t** h_ls){

//  printf("Allocating memory...\n");
    for(int k=0; k<K; ++k){ 
        // Host
        // Allocate memory for struct
        h_ls[k] = (layer_t*)malloc(sizeof(layer_t*));
        // Allocate memory for # of input neurons
        h_ls[k]->N = *(int*)malloc(sizeof(int));
        // Set # of input neurons
        h_ls[k]->N = N - k * (R-1);
        // Allocate memory for input array
        h_ls[k]->x = (double*)malloc(h_ls[k]->N * sizeof(double));
        // Set # of weights (R elements for each input neuron)
        int w_N = h_ls[k]->N * R;
        // Allocate memory for weights array
        h_ls[k]->W = (double*)malloc(w_N *sizeof(double));
        // Allocate memory for bias
        h_ls[k]->b = *(double*)malloc(sizeof(double));

    }

}

void generate_values(int K, layer_t** h_ls){

//    printf("Generating input values, weights and biases...\n");
    for(int k=0; k<K; ++k){
        // Host
        // Set # of weights (R elements for each input neuron)
        int w_N = h_ls[k]->N * R;
        // Fill weights
        fill_array(h_ls[k]->W, w_N);
        // Fill bias
        h_ls[k]->b = (double) rand()/RAND_MAX;

    }
    // Fill input array for first layer with random values in the range [0,1]
    fill_array(h_ls[0]->x, h_ls[0]->N);

}

void copy_to_device(int K, layer_t** h_ls, layer_t** d_ls){

    for(int k =0; k<K; ++k){
        // Allocate memory for # of input neurons
        cudaSafeCall(cudaMalloc((void **)&(d_ls[k]->N), sizeof(int)));
        // Allocate memory for input array
        cudaSafeCall(cudaMalloc((void **)&(d_ls[k]->x), h_ls[k]->N * sizeof(double)));
        // Allocate memory for weights array
        cudaSafeCall(cudaMalloc((void **)&(d_ls[k]->W), w_N *sizeof(double)));
        // Allocate memory for bias
        cudaSafeCall(cudaMalloc((void **)&(d_ls[k]->b), sizeof(double)));
        // Device
        // Fill weights
        cudaSafeCall(cudaMemcpy(d_ls[k]->N, h_ls[k]->N, sizeof(int), cudaMemcpyHostToDevice));
        // Fill weights
        cudaSafeCall(cudaMemcpy(d_ls[k]->W, h_ls[k]->W, w_N*sizeof(double), cudaMemcpyHostToDevice));
        // Fill bias
        cudaSafeCall(cudaMemcpy(d_ls[k]->b, h_ls[k]->b, sizeof(double), cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy(d_ls[0]->x, h_ls[0]->x, h_ls[0]->N*sizeof(double), cudaMemcpyHostToDevice));
    }
}

void generate_network(int K, int N, layer_t** h_ls, layer_t** d_ls){
// Allocate memory and populate layers weights and bias with random values in the range [0,1]
//    printf("Generating network...\n");
    // Allocate memory
    initialize_network(N, K, h_ls);
    // Generate values
    generate_values(K, h_ls);
    // Copy to device
    copy_to_device(K, h_ls, d_ls)

//    printf("Network generated.\n");
}
*/
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

__global__ void kernel(layer_t l, layer_t l_next){
// Kernel function: given layer (inputs, weights, bias),
// compute the activations

    // Matrix multiplication
    for(int i=0; i < l.N - R + 1; ++i){ // Loop over output neurons
        l_next.x[i] = l.b; // Initialize to bias
        for(int j=0; j < R; ++j){
            l_next.x[i] += (l.x[i + j] * l.W[i*R + j]); // MAC
        }
        activation(&(l_next.x[i]));
    }
}
__global__ void kernel2(layer_t l, double* y){
// Kernel function: given layer (inputs, weights, bias),
// compute the activations

    // Matrix multiplication
    for(int i=0; i < l.N - R + 1; ++i){ // Loop over output neurons
        y[i] = l.b; // Initialize to bias
        for(int j=0; j < R; ++j){
            y[i] += (l.x[i + j] * l.W[i*R + j]); // MAC
        }
        activation(&y[i]);
    }
}

// Define propagation function
void forward(layer_t* ls, int N, int K, double* output){
//  Compute activations, applying the kernel function
//  to inputs, weights and biases of each layer, thus obtaining
//  the activations which serve as input for the next one.

    // Loop over layers (except last one)
    for(int k=0; k < K-1; ++k){

        // Compute activations and store them as input for next layer
        kernel<<<(N+BLKDIM-1)/BLKDIM, BLKDIM>>>(ls[k], ls[k+1]);
        cudaCheckError();
        N = N - R + 1;
        
    }
    // Store last activations as output
    kernel2<<<(N+BLKDIM-1)/BLKDIM, BLKDIM>>>(ls[K-1], output);
    cudaCheckError();
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
    layer_t* h_ls[K]; // host copy
    layer_t* d_ls[K]; // device copy
/*    for(int k=0; k<K; k++){
        cudaMalloc((void **) &d_ls[k], sizeof(d_ls[k]));
    }
*/
	// Set the seed
    srand(42);
printf("Before\n");
	// Prepare the network: allocate memory, fill weights and bias, fill first layer's input
    generate_network(K, N, h_ls, d_ls);
printf("After\n");
/*
    // Allocate memory for last output
    size_t out_size = (N - K*(R-1))*sizeof(double);
    h_output = (double*)malloc(out_size);
    cudaSafeCall(cudaMalloc((void **)&d_output, out_size));
*/

/*
	// Set the start time
    tstart = hpc_gettime();
	// Forward pass: compute activations for each layer, storing the result as input for the next one
    forward(h_ls, N, K, d_output);
    // Sinchronize
    cudaDeviceSynchronize();
    // Set the stop time
    tstop = hpc_gettime();
    // Copy result to host
    cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost);
    // Print execution time
    printf("Execution time %.2f\n", tstop - tstart);
*/


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
