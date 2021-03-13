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
void write_layer(layer_t* l, char* filename){
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
void write_array(double* a, int n, char* filename){
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
void initialize_network(int K, int N, layer_t* ls[K]){

//	printf("Allocating memory...\n");
	for(int k=0; k<K; ++k){
        // Allocate memory for struct
        ls[k] = (layer_t*)malloc(sizeof(layer_t));
        // Allocate memory for # of input neurons
        ls[k]->N = *(int*)malloc(sizeof(int));
		// Set # of input neurons
    	ls[k]->N = N - k * (R-1);
    	// Allocate memory for input array
		ls[k]->x = (double*)malloc(ls[k]->N * sizeof(double));
		// Set # of weights (R elements for each output neuron)
    	int w_N = (ls[k]->N-R+1) * R;
		// Allocate memory for weights array
  		ls[k]->W = (double*)malloc(w_N *sizeof(double));
        // Allocate memory for bias
        ls[k]->b = *(double*)malloc(sizeof(double));
    }
}

void generate_values(int K, layer_t* ls[K]){

//    printf("Generating input values, weights and biases...\n");
    for(int k=0; k<K; ++k){
        // Set # of weights (R elements for each output neuron)
        int w_N = (ls[k]->N-R+1) * R;
        // Fill weights
        fill_array(ls[k]->W, w_N);
        // Fill bias
        ls[k]->b = (double) rand()/RAND_MAX;
    }
    // Fill input array for first layer with random values in the range [0,1]
    fill_array(ls[0]->x, ls[0]->N);
}

void generate_network(int K, int N, layer_t* ls[K]){
// Allocate memory and populate layers weights and bias with random values in the range [0,1]
//    printf("Generating network...\n");

    // Allocate memory
    initialize_network(K, N, ls);

    // Generate values
    generate_values(K, ls);

//    printf("Network generated.\n");
}


// Define activation function (sigmoid)
void activation(double* x){
	*x = 1/(1 + exp(-*x));
}

void kernel(double* x, double* W, double b, int N, double* y){
// Kernel function: given layer (inputs, weights, bias),
// compute the activations

    #pragma omp parallel for
    // Matrix multiplication
    for(int i=0; i < N - R + 1; ++i){ // Loop over output neurons
        y[i] = b; // Initialize to bias
        for(int j=0; j < R; ++j){
            y[i] += (x[i + j] * W[i*R + j]); // MAC
        }
        activation(&y[i]);
    }
    // Free useless memory
    free(x);
    free(W);
}

// Define propagation function
void forward(int K, layer_t* ls[K], double* output){
//  Compute activations, applying the kernel function
//  to inputs, weights and biases of each layer, thus obtaining
//  the activations which serve as input for the next one.

    // Loop over layers (except last one)
    for(int k=0; k < K-1; ++k){
        // Compute activations and store them as input for next layer
        kernel(ls[k]->x, ls[k]->W, ls[k]->b, ls[k]->N, ls[k+1]->x);
        
    }
    // Store last activations as output
    kernel(ls[K-1]->x, ls[K-1]->W, ls[K-1]->b, ls[K-1]->N, output);
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
    layer_t** ls = malloc(K * sizeof(layer_t*));

	// Set the seed
    srand(42);

	// Prepare the network: allocate memory, fill weights and bias, fill first layer's input
    generate_network(K, N, ls);

    // Allocate memory for last output
    double* output = (double*)malloc((N - K*R + K)*sizeof(double));

	// Set the start time
    tstart = hpc_gettime();
	// Forward pass: compute activations for each layer, storing the result as input for the next one
    forward(K, ls, output);
    // Set the stop time
    tstop = hpc_gettime();
    // Print execution time
    printf("Execution time %.2f\n", tstop - tstart);

    write_array(output, N- K*R + K, "omp.txt");

    return EXIT_SUCCESS;
}
