/****************************************************************************
 *
 * nn_cuda.cu - Neural Network evaluation exploiting NVIDIA CUDA
 *
 *
 * --------------------------------------------------------------------------
 *
 * Compile with:
 * nvcc nn_cuda.cu -o nn_cuda
 *
 * Run with:
 * ./nn_cuda [N K]
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

#define R 5
#define BLKDIM 1024
//#define NO_CUDA_CHECK_ERROR

// Define struct to store weights, bias and # of inputs for each layer
typedef struct {
	int N;
	double* W;
	double b;
} layer_t;

// Print array to screen
void print_array(double* a, int n){
    int i;
    printf("Values\n\n");
    for(i=0; i<n; ++i){
        printf("%lf\t", a[i]);
    }
    printf("\n");
}

// Write array to file
void write_array(double* a, int n, const char* filename){
    FILE* fptr = fopen(filename, "w");
    int i;
    fprintf(fptr, "Values\n\n");
    for(i=0; i<n; ++i){
        fprintf(fptr, "%lf\t", a[i]);
    }
    fclose(fptr);
}


// Fill given array with random values in the range [0,1]
void fill_array(double* array, int n){
    int i;
	for(i=0; i<n; ++i)
		array[i] = (double) rand() / RAND_MAX;
}

// Check GPU memory usage
void checkMemory(){
    size_t free_byte, total_byte;

    cudaError cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
    if (cudaSuccess != cuda_status){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
    }

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage:\ntotal = %.2f MB, used = %.2f, free = %.2f MB\n",
        total_db/1024.0/1024.0, used_db/1024.0/1024.0, free_db/1024.0/1024.0);
}

// Define activation function (sigmoid)
__device__ void activation(double* x){
	*x = 1/(1 + exp(-(*x)));
}

__global__ void kernel(const layer_t l, double* out, double* in){
// Kernel function: given layer (inputs, weights, bias), compute the activations
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int id = bx*blockDim.x + tx;

    int j;
    if(id < l.N-R+1){ // Control over size of output
        out[id] = l.b; // Initialize to bias
        for(j=0; j < R; ++j){ // Loop over input neurons
            out[id] += (in[id + j] * l.W[id*R + j]); // MAC
        }
        activation(&(out[id])); // apply activation function
    }
}

// Define propagation function
void forward(const int N, const int K, layer_t* ls, double* d_output, double* d_input){
//  Compute activations, sequentially applying the kernel function
//  to inputs, weights and biases of each layer, thus obtaining
//  the activations which serve as input for the next one.
    int k, n;
    size_t input_size;
    // Loop over layers
    for(k=0; k < K; ++k){
        n = N - (k+1)*(R-1); // # of output neurons
        input_size = n*sizeof(double);
        // Compute activations and store them as input for next layer
        kernel<<<(n+BLKDIM-1)/(BLKDIM), BLKDIM>>>(ls[k], d_output, d_input);
        cudaCheckError();
        if(k<K-1) cudaSafeCall(cudaMemcpy(d_input, d_output, input_size, cudaMemcpyDeviceToDevice));
    }
}

int main(int argc, char* argv[])
{
    int N = 500000; // size of the first layer
    int K = 150; // number of layers
    double tstart, tstop; // timing variables
    double* h_input, *h_output; // host
    double* d_output, *d_input; // device

    if(argc>1) {    
        N = atoi(argv[1]);
        K = atoi(argv[2]);
        if(N-K*R+K<1){ // Control over size of last layer
            printf("Invalid size for last output:%d.\n", N-K*R+K);
            return EXIT_FAILURE;
        }
    }
    
	// Instantiate a struct for each layer to store weights and bias
    layer_t ls[K];
    double* W[K]; // weights

    // Allocate memory
    size_t input_size = N*sizeof(double); // inputs for first layer
    size_t out_size1 = (N-R+1)*sizeof(double); // activations for first layer 
    size_t out_sizeK = (N-K*R+K)*sizeof(double); // activations for last layer
    h_input = (double*)malloc(input_size);
    h_output = (double*)malloc(out_sizeK); // to store activations for last layer
    cudaSafeCall(cudaMalloc((void **)&d_output, out_size1)); // to store activations for each layer
    cudaSafeCall(cudaMalloc((void **)&d_input, input_size)); // to store inputs for each layer

	// Set the seed
    srand(42);

	// Prepare the network: allocate memory, fill weights and bias, fill first layer's input
    int k, n, W_n;
    for(k=0; k<K; ++k){
        n = N - k * (R-1); // # of inputs
        W_n = (n-R+1)*R; // # of weights (R elements for each output neuron)
        // Allocate memory for weights
        W[k] = (double*)malloc(W_n*sizeof(double)); // Host
        cudaSafeCall(cudaMalloc((void **)&(ls[k].W), W_n*sizeof(double))); // Device
        // Fill values
        // Host
        ls[k].N = n; // # of input neurons
        if(!k) fill_array(h_input, n); // first input
        fill_array(W[k], W_n); // weights
        ls[k].b = (double) rand()/RAND_MAX; // bias
        // Device
        if(!k) cudaSafeCall(cudaMemcpy(d_input, h_input, n*sizeof(double), cudaMemcpyHostToDevice)); // first input
        cudaSafeCall(cudaMemcpy(ls[k].W, W[k], W_n*sizeof(double), cudaMemcpyHostToDevice)); // weights
    }

	// Set the start time
    tstart = hpc_gettime();
	// Forward pass: compute activations for each layer
    forward(N, K, ls, d_output, d_input);
    // Sinchronize
    cudaSafeCall(cudaDeviceSynchronize());
    // Set the stop time
    tstop = hpc_gettime();
    // Copy result to host
    cudaMemcpy(h_output, d_output, out_sizeK, cudaMemcpyDeviceToHost);
    // Print execution time
    printf("Execution time %.4lf\n", tstop - tstart);

    // Write last activations to file
    write_array(h_output, N-K*R+K, "cuda.txt");
    
    // Cleanup
    free(h_input); free(h_output); 
    cudaSafeCall(cudaFree(d_input)); cudaSafeCall(cudaFree(d_output));
    
    for(int k=0; k<K; ++k){
        free(W[k]);
        cudaSafeCall(cudaFree(ls[k].W));
    }

    return EXIT_SUCCESS;
}
