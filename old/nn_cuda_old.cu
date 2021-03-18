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

// Print inputs, weights and bias for given layer to screen
void print_layer(int k, long N, float* x, float* W, float* b){
    // Get indices to retrieve elements from arrays
    long x_start=0, x_stop=N, W_start=0, W_stop=(N-R+1)*R;
    for(int i=1; i<k+1; ++i){
        x_start = x_stop;
        W_start = W_stop;
        x_stop += (N - (i)*(R-1));
        W_stop += (N - (i+1)*(R-1))*R;
    }

    printf("inputs:\n");
    for(long i=x_start; i<x_stop; ++i){
        printf("%lf\t", x[i]);
    }
    printf("\n");
    printf("weights:\n");
    for(long i=W_start; i<W_stop; ++i){
        printf("%lf\t", W[i]);
    }
    printf("\n");
    printf("bias:\n");
    printf("%lf\n", b[k]);

}

// Write inputs, weights and bias for given layer to file
void write_layer(int k, long N, float* x, float* W, float* b, const char* filename){
    FILE* fptr = fopen(filename, "w");
    // Get indices to retrieve elements from arrays
    long x_start=0, x_stop=N, W_start=0, W_stop=(N-R+1)*R;
    for(int i=1; i<k+1; ++i){
        x_start = x_stop;
        W_start = W_stop;
        x_stop += (N - (i)*(R-1));
        W_stop += (N - (i+1)*(R-1))*R;
    }
    fprintf(fptr, "Inputs\n\n");
    for(long i=x_start; i<x_stop; ++i){
        fprintf(fptr, "%lf\t", x[i]);
    }
    fprintf(fptr, "\n\n");
    fprintf(fptr, "Weights\n\n");
    for(long i=W_start; i<W_stop; ++i){
        fprintf(fptr, "%lf\t", W[i]);
    }
    fprintf(fptr, "\n\n");
    fprintf(fptr, "Bias\n\n");
    fprintf(fptr, "%lf\n", b[k]);
    fclose(fptr);
}

// Print array to screen
void print_array(float* a, long n){

    printf("Values\n\n");
    for(long i=0; i<n; ++i){
        printf("%lf\t", a[i]);
    }
    printf("\n");
}

// Write array to file
void write_array(float* a, long n, const char* filename){
    FILE* fptr = fopen(filename, "w");

    fprintf(fptr, "Results\n\n");
    for(long i=0; i<n; ++i){
        fprintf(fptr, "%lf\t", a[i]);
    }
    fclose(fptr);
}


// Fill given array with random values in the range [0,1]
void fill_array(float* array, long n){

    for(long i=0; i<n; ++i)
        array[i] = (float) rand() / RAND_MAX;
}

// Fill given array with random values in the range [0,1]
// until stop index, then fill with 0s.
void partial_fill_array(float* array, long n, long stop){

    for(long i=0; i<stop; ++i)
        array[i] = (float) rand() / RAND_MAX;
    for(long i=stop; i<n; ++i)
        array[i] = (float)0;
}

/*
__global__ void test(float* x){
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    x[index] = (float)index;
}


__device__ void myadd(float* x){
    *x = *x+1;
}

__global__ void kk(float* y, int k){
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    printf("I am thread #%d at it.#%d\n", id, k);
    myadd(&(y[k]));
}

*/
void checkMemory(){
    size_t free_byte, total_byte ;

    cudaError cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
    if (cudaSuccess != cuda_status){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
    }

    float free_db = (float)free_byte ;
    float total_db = (float)total_byte ;
    float used_db = total_db - free_db ;
    printf("GPU memory usage:\nused = %.2f, free = %.2f MB, total = %.2f MB\n",
        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

// Define activation function (sigmoid)
__device__ void activation(float* x){
    *x = 1/(1 + exp(-*x));
}

__global__ void kernel(int k, long N, long x_start, long x_stop, long W_start, float* d_x, float* d_W, float* d_b){
// Kernel function: compute the activations given inputs, weights and bias

    const int bx = blockIdx.x; //by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    const long id = bx*blockDim.y*blockDim.x + ty*blockDim.x + tx;
//    printf("I am thread #%d\n", id);

    if(id < N-R+1){
        d_x[x_stop + id] = d_b[k]; // Initialize to bias
        for(int j=0; j < R; ++j){
            d_x[x_stop + id] += (d_x[x_start + id + j] * d_W[W_start + id*R + j]); // MAC
        }
        activation(&(d_x[x_stop + id]));
    }
}

// Define propagation function
void forward(long N, int K, float* d_x, float* d_W, float* d_b, float* d_y){
//  Compute activations, applying the kernel function
//  to inputs, weights and biases of each layer, thus obtaining
//  the activations which serve as input for the next one.

    // Compute indices to retrieve each layer's input and weights
    long x_start[K], x_stop[K], W_start[K], W_stop[K];
    x_start[0]=0; x_stop[0]=N; W_start[0]=0; W_stop[0]=(N-R+1)*R;
    for(int i=1; i<K+1; ++i){
        x_start[i] = x_stop[i-1];
        W_start[i] = W_stop[i-1];
        x_stop[i] = x_stop[i-1] + (N - (i)*(R-1));
        W_stop[i] = W_stop[i-1] + (N - (i+1)*(R-1))*R;
    }

    // Loop over layers
    for(int k=0; k < K; ++k){
        // Compute activations and store them as input for next layer
        //long blocks = (N-R+1+BLKDIM-1)/BLKDIM;
        //kernel<<<blocks, BLKDIM>>>(k, N, x_start[k], x_stop[k], W_start[k], d_x, d_W, d_b);
        dim3 grid((N-R+1+BLKDIM*BLKDIM-1)/(BLKDIM*BLKDIM));
        dim3 block(BLKDIM, BLKDIM);
//        printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
        kernel<<<grid, block>>>(k, N, x_start[k], x_stop[k], W_start[k], d_x, d_W, d_b);
        cudaCheckError();
//        if(k<K-1){
//            cudaSafeCall(cudaMemcpy(d_x+x_stop[k], d_y, (N-R+1)*sizeof(float), cudaMemcpyDeviceToDevice));
//        }
        N = N-R+1;
    }
    printf("Copying from x+%li to x+%li\n", x_start[K], x_stop[K]);
    cudaSafeCall(cudaMemcpy(d_y, d_x+x_start[K], N*sizeof(float), cudaMemcpyDeviceToDevice));
}

// Define activation function (sigmoid)
__device__ float act(float x){
    return 1/(1 + exp(-x));
}

__global__ void kernel_shared(int k, long N, long x_start, long x_stop, long W_start, float* d_x, float* d_W, float* d_b){
// Kernel function: compute the activations given inputs, weights and bias
    const int bx = blockIdx.x; //by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    const long id = bx*blockDim.y*blockDim.x + ty*blockDim.x + tx;
    float psum = 0;
//    printf("I am thread #%d\n", id);
/*
    __shared__ float b;
    if(!(ty*blockDim.x + tx)) b = d_b[k];
    __syncthreads();
*/
    if(id < N-R+1){
        psum = d_b[k]; // Initialize to bias
        for(int j=0; j < R; ++j){
            psum += (d_x[x_start + id + j] * d_W[W_start + id*R + j]); // MAC
        }
        d_x[x_stop + id] = act(psum);
    }
}

// Define propagation function
void forward_shared(long N, int K, float* d_x, float* d_W, float* d_b, float* d_y){
//  Compute activations, applying the kernel function
//  to inputs, weights and biases of each layer, thus obtaining
//  the activations which serve as input for the next one.

    // Compute indices to retrieve each layer's input and weights
    long x_start[K], x_stop[K], W_start[K], W_stop[K];
    x_start[0]=0; x_stop[0]=N; W_start[0]=0; W_stop[0]=(N-R+1)*R;
    for(int i=1; i<K+1; ++i){
        x_start[i] = x_stop[i-1];
        W_start[i] = W_stop[i-1];
        x_stop[i] = x_stop[i-1] + (N - (i)*(R-1));
        W_stop[i] = W_stop[i-1] + (N - (i+1)*(R-1))*R;
    }

    // Loop over layers
    for(int k=0; k < K; ++k){
        // Compute activations and store them as input for next layer
        //long blocks = (N-R+1+BLKDIM-1)/BLKDIM;
        //kernel<<<blocks, BLKDIM>>>(k, N, x_start[k], x_stop[k], W_start[k], d_x, d_W, d_b);
        dim3 grid((N-R+1+BLKDIM*BLKDIM-1)/(BLKDIM*BLKDIM));
        dim3 block(BLKDIM, BLKDIM);
//        printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
        kernel_shared<<<grid, block>>>(k, N, x_start[k], x_stop[k], W_start[k], d_x, d_W, d_b);
        cudaCheckError();
//        if(k<K-1){
//            cudaSafeCall(cudaMemcpy(d_x+x_stop[k], d_y, (N-R+1)*sizeof(float), cudaMemcpyDeviceToDevice));
//        }
        N = N-R+1;
    }
    cudaSafeCall(cudaMemcpy(d_y, d_x+x_start[K], N*sizeof(float), cudaMemcpyDeviceToDevice));
}

int main(int argc, char* argv[])
{
    
    long N = 500000; // size of the first layer
    int K = 100; // number of layers
    long double tstart, tstop; // timing variables
    float* h_x, *h_W, *h_b, *h_y; // host copies
    float* d_x, *d_W, *d_b, *d_y; // device copies

    if(argc>1) {    
        N = atoi(argv[1]);
        K = atoi(argv[2]);
    }

	// Prepare the network: allocate memory, fill weights and bias, fill first layer's input    
    // GENERATE NETWORK

    // Set arrays sizes
    long x_size=N, W_size=(N-R+1)*R, b_size=K, y_size=N-K*R+K;
    for(int k=1; k<K; ++k){
        x_size += (N - (k)*(R-1));
        W_size += (N - (k+1)*(R-1))*R;
    }
    x_size += y_size; // To store final output

    // Host
    // Memory allocation
    h_x = (float*)malloc(x_size * sizeof(float)); // Input
    h_W = (float*)malloc(W_size * sizeof(float)); // Weights
    h_b = (float*)malloc(b_size * sizeof(float)); // Bias
    h_y = (float*)malloc(y_size * sizeof(float)); // Output

    // Fill with random numbers
    srand(42); // to allow replicability
    // Only first N input values must be filled (others set to 0)
    partial_fill_array(h_x, x_size, N); // Input
    fill_array(h_W, W_size); // Weights
    fill_array(h_b, b_size); // Bias


    // Device
    // Memory allocation
    cudaSafeCall(cudaMalloc((void **)&d_x, x_size * sizeof(float))); // Input
    cudaSafeCall(cudaMalloc((void **)&d_W, W_size * sizeof(float))); // Weights
    cudaSafeCall(cudaMalloc((void **)&d_b, b_size * sizeof(float))); // Bias
    cudaSafeCall(cudaMalloc((void **)&d_y, y_size * sizeof(float))); // Output

    // Copy from host
    cudaSafeCall(cudaMemcpy(d_x, h_x, x_size * sizeof(float), cudaMemcpyHostToDevice)); // Input
    cudaSafeCall(cudaMemcpy(d_W, h_W, W_size * sizeof(float), cudaMemcpyHostToDevice)); // Weights
    cudaSafeCall(cudaMemcpy(d_b, h_b, b_size * sizeof(float), cudaMemcpyHostToDevice)); // Bias
    cudaSafeCall(cudaMemset(d_y, 0, y_size * sizeof(float))); // Output
    
    printf("x_size\tW_size\tb_size\ty_size\n");
    printf("%li\t%li\t%li\t%li\n", x_size, W_size, b_size, y_size);
    
    // Check GPU memory usage
    printf("Memory allocated, ");
    checkMemory();


    // PERFORM FORWARD PASS

	// Set the start time
    tstart = hpc_gettime();
	// Forward pass: compute activations for each layer, storing the result as input for the next one
    forward(N, K, d_x, d_W, d_b, d_y);
    // Sinchronize
    cudaDeviceSynchronize();
    // Set the stop time
    tstop = hpc_gettime();
    // Print execution time
    printf("Execution time %.4Lf\n", tstop - tstart);

    // Copy result to host
    cudaMemcpy(h_y, d_y, y_size*sizeof(float), cudaMemcpyDeviceToHost);

    // Write result to file
    //write_array(h_y, y_size, "cuda.txt");

    // Set the start time
    tstart = hpc_gettime();
    // Forward pass: compute activations for each layer, storing the result as input for the next one
    forward_shared(N, K, d_x, d_W, d_b, d_y);
    // Sinchronize
    cudaDeviceSynchronize();
    // Set the stop time
    tstop = hpc_gettime();
    // Print execution time
    printf("Execution time %.4Lf\n", tstop - tstart);

    // Copy result to host
    cudaMemcpy(h_y, d_y, y_size*sizeof(float), cudaMemcpyDeviceToHost);

    // Write result to file
    write_array(h_y, y_size, "cuda.txt");

    // Clean up
    free(h_x); free(h_W); free(h_b); free(h_y);
    cudaFree(d_x); cudaFree(d_W); cudaFree(d_b); cudaFree(d_y);

    // Check GPU memory usage
    printf("Memory freed, ");
    checkMemory();


/*  TEST
    float* out = (float*)malloc(N*sizeof(float));
    float* d_out;
    cudaMalloc((void **)&d_out,N*sizeof(float));
    kernel1<<<(N+BLKDIM-1)/BLKDIM, BLKDIM>>>(d_out);
    // Sinchronize
    cudaDeviceSynchronize();
    // Set the stop time
    tstop = hpc_gettime();
    // Copy result to host
    cudaMemcpy(out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    // Print execution time
    printf("Execution time %.2f\n", tstop - tstart);
    write_array(out, N, "cuda.txt");
*/

    return EXIT_SUCCESS;
}
