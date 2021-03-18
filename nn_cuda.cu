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


#define BLKDIM 32
#define R 3

// Define struct to store inputs, weights and bias for each layer
typedef struct {
	int N;
	double* x;
	double* W;
	double b;
} layer_t;

// Print inputs, weights and bias for given layer to screen
void print_layer(layer_t l){
	printf("inputs:\n");
	for(int i=0; i<l.N; ++i){
		printf("%lf\t", l.x[i]);
	}
	printf("\n");
	printf("weights:\n");
	for(int i=0; i<(l.N-R+1) * R; ++i){
		printf("%lf\t", l.W[i]);
	}
	printf("\n");
	printf("bias:\n");
	printf("%lf\n", l.b);
}

// Write inputs, weights and bias for given layer to file
void write_layer(layer_t l, const char* filename){
    FILE* fptr = fopen(filename, "w");

    fprintf(fptr, "Inputs\n\n");
    for(int i=0; i<l.N; ++i){
        fprintf(fptr, "%lf\t", l.x[i]);
    }
    fprintf(fptr, "\n\n");
    fprintf(fptr, "Weights\n\n");
    for(int i=0; i<(l.N-R+1) * R; ++i){
        fprintf(fptr, "%lf\t", l.W[i]);
    }
    fprintf(fptr, "\n\n");
    fprintf(fptr, "Bias\n\n");
    fprintf(fptr, "%lf\n", l.b);
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

// Check GPU memory usage
void checkMemory(){
    size_t free_byte, total_byte ;

    cudaError cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
    if (cudaSuccess != cuda_status){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
    }

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage:\nused = %.2f, free = %.2f MB, total = %.2f MB\n",
        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}


// Define activation function (sigmoid)
__device__ void activation(double* x){
	*x = 1/(1 + exp(-*x));
}

__global__ void kernel(layer_t l, double* out){
// Kernel function: given layer (inputs, weights, bias),
// compute the activations
    const int bx = blockIdx.x; //by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    const int id = bx*blockDim.y*blockDim.x + ty*blockDim.x + tx;

    // Matrix multiplication
    if(id < l.N-R+1){
        out[id] = l.b; // Initialize to bias
        for(int j=0; j < R; ++j){
            out[id] += (l.x[id + j] * l.W[id*R + j]); // MAC
        }
        activation(&(out[id]));
    }
}

// Define propagation function
void forward(int N, int K, layer_t* ls, double* d_output){
//  Compute activations, applying the kernel function
//  to inputs, weights and biases of each layer, thus obtaining
//  the activations which serve as input for the next one.
    int n;
    // Loop over layers
    for(int k=0; k < K; ++k){
        n = N - k * (R-1);
        dim3 grid((N-R+1+BLKDIM*BLKDIM-1)/(BLKDIM*BLKDIM));
        dim3 block(BLKDIM, BLKDIM);
        // Compute activations and store them as input for next layer
        kernel<<<grid, block>>>(ls[k], d_output);
        cudaCheckError();
        if(k<K-1) cudaSafeCall(cudaMemcpy(ls[k+1].x, d_output, (n-R+1)*sizeof(double), cudaMemcpyDeviceToDevice));
        // Free useless memory
        //cudaFree(d_ls[k].x);
        //cudaFree(d_ls[k].W);
    }
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

    layer_t ls[K]; // host
    double* x[K];
    double* W[K];

    // Allocate memory for output
    size_t out_size1 = (N-R+1)*sizeof(double); // activations for first layer 
    size_t out_sizeK = (N-K*R+K)*sizeof(double); // last layer
    h_output = (double*)malloc(out_sizeK); 
    cudaSafeCall(cudaMalloc((void **)&d_output, out_size1));


	// Set the seed
    srand(42);

// GENERATE NETWORK
	// Prepare the network: allocate memory, fill weights and bias, fill first layer's input
    int n, W_n;
    for(int k=0; k<K; ++k){
        n = N - k * (R-1); // # of inputs
        W_n = (n-R+1)*R; // # of weights (R elements for each output neuron)

        // Allocate memory
        // Host
        x[k] = (double*)malloc(n * sizeof(double)); // input
        W[k] = (double*)malloc(W_n * sizeof(double)); // weights
        // Device
        cudaSafeCall(cudaMalloc((void **)&(ls[k].x), n * sizeof(double))); // input
        cudaSafeCall(cudaMalloc((void **)&(ls[k].W), W_n * sizeof(double))); // weights
        
        // Fill values
        // Host
        ls[k].N = n; // # of input neurons
        if(!k) fill_array(x[k], n); // first input
        fill_array(W[k], W_n); // weights
        ls[k].b = (double) rand()/RAND_MAX; // bias
        // Device
        if(!k) cudaSafeCall(cudaMemcpy(ls[k].x, x[k], n * sizeof(double), cudaMemcpyHostToDevice)); // first input
        cudaSafeCall(cudaMemcpy(ls[k].W, W[k], W_n * sizeof(double), cudaMemcpyHostToDevice)); // weights

    }

//    print_array(x[0], N);
//    print_array(W[0], (N-R+1)*R);

//    // Check GPU memory usage
//    printf("Memory allocated, ");
//    checkMemory();

	// Set the start time
    tstart = hpc_gettime();
	// Forward pass: compute activations for each layer, storing the result as input for the next one
    forward(N, K, ls, d_output);

    // Sinchronize
    cudaDeviceSynchronize();
    // Set the stop time
    tstop = hpc_gettime();
    // Copy result to host
    cudaMemcpy(h_output, d_output, out_sizeK, cudaMemcpyDeviceToHost);
    // Print execution time
    printf("Execution time %.4f\n", tstop - tstart);

    // Write last activations to file
    write_array(h_output, N-K*R+K, "cuda.txt");

    free(h_output); cudaFree(d_output);

    for(int k=0; k<K; ++k){
        free(x[k]);
        free(W[k]);
        cudaFree(ls[k].x);
        cudaFree(ls[k].W);
    }

//    // Check GPU memory usage
//    printf("Memory allocated, ");
//    checkMemory();




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
