/****************************************************************************
 *
 * project.c - Neural Network evaluation
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
 * gcc -fopenmp project.c -o project
 *
 * Run with:
 * ./project
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include "hpc.h"

#define R 5

// Define struct to store inputs, weights and bias for each layer
typedef struct layer{
	int N;
	double* x;
	double* W;
	double b;
} layer;

// Print inputs, weights and bias for given layer to screen
void print_layer(layer l){
	printf("inputs:\n");
	for(int i=0; i<l.N; ++i){
		printf("%lf\t", l.x[i]);
	}
	printf("\n");
	printf("weights:\n");
	for(int i=0; i<l.N * R; ++i){
		printf("%lf\t", l.W[i]);
	}
	printf("\n");
	printf("bias:\n");
	printf("%lf\n", l.b);
}

// Fill given array with random values in the range [0,1]
void fill_array(double* array, int n){

	for(int i=0; i<n; ++i)
		array[i] = (double) rand() / RAND_MAX;
}

// Allocate memory to store input and weight arrays of each layer
void initialize_network(layer* ls, int N, int K){

//	printf("Allocating memory...\n");
	for(int k=0; k<K; ++k){
		// Set # of input neurons
    	ls[k].N = N - k * (R-1);
    	// Allocate memory for input array
		ls[k].x = (double*)malloc(ls[k].N * sizeof(double));
		// Set # of weights (R elements for each input neuron)
    	int w_N = ls[k].N * R;
		// Allocate memory for weights array
  		ls[k].W = (double*)malloc(w_N *sizeof(double));
    }

}

void generate_values(layer* ls, int N, int K){

//    printf("Generating input values, weights and biases...\n");
    for(int k=0; k<K; ++k){
        // Set # of input neurons
        ls[k].N = N - k * (R-1);
        // Set # of weights (R elements for each input neuron)
        int w_N = ls[k].N * R;
        // Fill weights
        fill_array(ls[k].W, w_N);
        // Fill bias
        ls[k].b = (double) rand()/RAND_MAX;
    }
    // Fill input array for first layer with random values in the range [0,1]
    fill_array(ls[0].x, ls[0].N);
}

void generate_network(layer* ls, int N, int K){
// Allocate memory and populate layers weights and bias with random values in the range [0,1]
//    printf("Generating network...\n");
    // Allocate memory
    initialize_network(ls, N, K);
    // Generate values
    generate_values(ls, N, K);

//    printf("Network generated.\n");
}


// Define activation function (sigmoid)
double activation(double x){
	return 1/(1 + exp(-x));
}

void kernel(layer l, double* y){
// Kernel function: given layer (inputs, weights, bias),
// compute the activations

	#pragma omp parallel for
	// Matrix multiplication
	for(int i=0; i < l.N - R + 1; ++i){ // Loop over output neurons
		y[i] = l.b; // Initialize to bias
		for(int j=0; j < R; ++j){
			y[i] += (l.x[i + j] * l.W[i*R + j]); // MAC
		}
		y[i] = activation(y[i]);
	}
}

// Define propagation function
void forward(layer* ls, int K, double* output){
//  Compute activations, applying the kernel function
//	to inputs, weights and biases of each layer, thus obtaining
//	the activations which serve as input for the next one.

	// Loop over layers (except last one)
	for(int k=0; k < K-1; ++k){
		// Compute activations and store them as input for next layer
		kernel(ls[k], ls[k+1].x);
		
	}
	// Store last activations as output
	kernel(ls[K-1], output);
}


// Write inputs, weights and bias for given layer to file
void write_layer(layer l, char* filename){
    FILE* fptr = fopen(filename, "w");

    fprintf(fptr, "Inputs\n\n");
    for(int i=0; i<l.N; ++i){
        fprintf(fptr, "%lf\t", l.x[i]);
    }
    fprintf(fptr, "\n\n");
    fprintf(fptr, "Weights\n\n");
    for(int i=0; i<l.N * R; ++i){
        fprintf(fptr, "%lf\t", l.W[i]);
    }
    fprintf(fptr, "\n\n");
    fprintf(fptr, "Bias\n\n");
    fprintf(fptr, "%lf\n", l.b);
}

int main(int argc, char* argv[])
{
	
	// Declare user input variables.
	// N is the size of the first layer, K is the number of layers.
	
	int N,K;

	// Check if values were correctly provided
    if(argc>1) {	
    	N = atoi(argv[1]);
    	K = atoi(argv[2]);
//    	printf("The network consists of %d layers. Layer 0 has size %d. R is %d.\n", K,N,R);

    }
    else {
    	printf("Please provide the dimension of the first layer (N) and the number of layers (K)");
    	return EXIT_FAILURE;
    }

    // Declare timing variables
    double tstart, tstop;

	// Instantiate a struct for each layer to store inputs, weights and bias
	// (plus one to store the final output)
    layer ls[K];

	// Set the seed
    srand(42);

	// Prepare the network: allocate memory, fill weights and bias, fill first layer's input
    generate_network(ls, N, K);

    // Allocate memory for last output
    int n = N;
    for (int k=0; k<K; k++)
    	n = n - R + 1;
    double* output = (double*)malloc(n*sizeof(double));

//	printf("Performing forward pass...\n");
	// Set the start time
    tstart = hpc_gettime();
	// Forward pass: compute activations for each layer, storing the result as input for the next one
    forward(ls, K, output);
    // Set the stop time
    tstop = hpc_gettime();
//    printf("Forward pass performed.\n");
    // Print execution time
    printf("Execution time %.2f\n", tstop - tstart);


/*  
	printf("Layer #1:");
    print_layer(ls[0]);
    write_layer(ls[0], "layer0.txt");
    printf("Layer #%d:",K+1);
    print_layer(ls[K]);
    write_layer(ls[K], "layerK.txt");
*/

    return EXIT_SUCCESS;
}
