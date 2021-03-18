/****************************************************************************
 *
 * nn_omp.c - Neural Network evaluation exploiting OpenMP
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
 * gcc -fopenmp nn_omp.c -o nn_omp
 *
 * Run with:
 * ./nn_omp N K
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

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


// Define activation function (sigmoid)
void activation(double* x){
	*x = 1/(1 + exp(-*x));
}

void kernel(layer_t l, double* out){
// Kernel function: given layer (inputs, weights, bias),
// compute the activations

    #pragma omp parallel for
    // Matrix multiplication
    for(int i=0; i < l.N-R+1; ++i){ // Loop over output neurons
        out[i] = l.b; // Initialize to bias
        for(int j=0; j < R; ++j){
            out[i] += (l.x[i + j] * l.W[i*R + j]); // MAC
        }
        activation(&(out[i]));
    }
}

// Define propagation function
void forward(layer_t* ls, int K, double* output){
//  Compute activations, applying the kernel function
//  to inputs, weights and biases of each layer, thus obtaining
//  the activations which serve as input for the next one.

    // Loop over layers (except last one)
    for(int k=0; k < K-1; ++k){
        // Compute activations and store them as input for next layer
        kernel(ls[k], ls[k+1].x);
        
    }
    // Store last activations as output
    kernel(ls[K-1], output);
}

int main(int argc, char* argv[])
{
	
	int N = 500000; // size of the first layer
    int K = 100; // number of layers
    double tstart, tstop; // timing variables

    if(argc>1) {	
    	N = atoi(argv[1]);
    	K = atoi(argv[2]);
    }

	// Instantiate a struct for each layer to store inputs, weights and bias
	// (plus one to store the final output)
    layer_t ls[K];

    // Allocate memory for output
    size_t out_size = (N-K*R+K)*sizeof(double); // activations for last layer 
    double* output = (double*)malloc(out_size); 
    
	// Set the seed
    srand(42);

// GENERATE NETWORK
	// Prepare the network: allocate memory, fill weights and bias, fill first layer's input
    int n, W_n;
    for(int k=0; k<K; ++k){
        n = N - k * (R-1); // # of inputs
        W_n = (n-R+1)*R; // # of weights (R elements for each output neuron)

        // Allocate memory
        ls[k].x = (double*)malloc(n * sizeof(double)); // input
        ls[k].W = (double*)malloc(W_n * sizeof(double)); // weights

        // Fill values
        ls[k].N = n; // # of input neurons
        if(!k) fill_array(ls[k].x, n); // first input
        fill_array(ls[k].W, W_n); // weights
        ls[k].b = (double) rand()/RAND_MAX; // bias
    }

//    print_array(ls[0].x, N);
//    print_array(ls[0].W, (N-R+1)*R);

	// Set the start time
    tstart = hpc_gettime();
	// Forward pass: compute activations for each layer, storing the result as input for the next one
    forward(ls, K, output);
    // Set the stop time
    tstop = hpc_gettime();
    // Print execution time
    printf("Execution time %.4f\n", tstop - tstart);

    // Write last activations to file
    write_array(output, N-K*R+K, "omp.txt");

    return EXIT_SUCCESS;
}
