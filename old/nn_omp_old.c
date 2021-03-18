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

// Define activation function (sigmoid)
void activation(float* x){
    *x = 1/(1 + exp(-*x));
}

void kernel(int k, long N, long x_start, long x_stop, long W_start, float* x, float* W, float* b){
// Kernel function: compute the activations given inputs, weights and bias
//    printf("Inside kernel\n");
//    printf("x_idx\tW_idx\tx_idx\n");
    #pragma omp parallel for
    for(long i=0; i < N-R+1; ++i){
//        printf("output index:%li\n", x_stop+i);
        x[x_stop + i] = b[k]; // Initialize to bias
        for(int j=0; j < R; ++j){
//            printf("%d\t%d\t%d\n", x_start + i + j, W_start + i*R + j, x_stop + i);
            x[x_stop + i] += (x[x_start + i + j] * W[W_start + i*R + j]); // MAC
        }
        activation(&(x[x_stop + i]));
    }
}

// Define propagation function
void forward(long N, int K, float* x, float* W, float* b, float* y){
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
//    printf("N=%li\tx_start[k]=%li\tx_stop[k]=%li\tW_start[k]=%li\tW_stop[k]=%li\n", N, x_start[0], x_stop[0], W_start[0], W_stop[0]);
    // Loop over layers
    for(int k=0; k < K; ++k){
//        printf("It#%d\n", k);
//        printf("k=%d\tN=%li\tx_start[k]=%li\tx_stop[k]=%li\tW_start[k]=%li\n", k, N, x_start[k], x_stop[k], W_start[k]);
        // Compute activations and store them as input for next layer
        kernel(k, N, x_start[k], x_stop[k], W_start[k], x, W, b);
        N = N-R+1;
    }
    memcpy(y, x+x_start[K], N*sizeof(float));
}


int main(int argc, char* argv[])
{
    
    long N = 500000; // size of the first layer
    int K = 100; // number of layers
    long double tstart, tstop; // timing variables
    float* x, *W, *b, *y; // host copies

    if(argc>1) {    
        N = atoi(argv[1]);
        K = atoi(argv[2]);
    }

    // Prepare the network: allocate memory, fill weights and bias, fill first layer's input    
    // GENERATE NETWORK

    // Set arrays sizes
    long x_size=N, W_size=(N-R+1)*R, b_size=K, y_size=N-K*R+K;
;
    for(int k=1; k<K; ++k){
        x_size += (N - (k)*(R-1));
        W_size += (N - (k+1)*(R-1))*R;
    }
    x_size += y_size; // To store final output


    // Host
    // Memory allocation
    x = (float*)malloc(x_size * sizeof(float)); // Input
    W = (float*)malloc(W_size * sizeof(float)); // Weights
    b = (float*)malloc(b_size * sizeof(float)); // Bias
    y = (float*)malloc(y_size * sizeof(float)); // Output

    // Fill with random numbers
    srand(42); // to allow replicability
    // Only first N input values must be filled (others set to 0)
    partial_fill_array(x, x_size, N); // Input
    fill_array(W, W_size); // Weights
    fill_array(b, b_size); // Bias

    // PERFORM FORWARD PASS

    // Set the start time
    tstart = hpc_gettime();
    // Forward pass: compute activations for each layer, storing the result as input for the next one
    forward(N, K, x, W, b, y);
    // Set the stop time
    tstop = hpc_gettime();
    // Print execution time
    printf("Execution time %.4Lf\n", tstop - tstart);

    // Print result to screen
//    print_array(y, y_size);

    // Write result to file
    write_array(y, y_size, "omp.txt");

    // Clean up
    free(x); free(W); free(b); free(y);

    return EXIT_SUCCESS;
}
