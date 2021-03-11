// Define propagation function
void forward_1(layer* ls, int K){
//  Compute activations, applying the forward function
//	to inputs, weights and biases of each layer, thus obtaining
//	the activations which serve as input for the next one.

	for(int k=0; k < K; ++k){
//        printf("\tLayer #%d\n", k+1);
        //#pragma omp parallel for
		// Matrix multiplication
		for(int i=0; i < ls[k+1].N; ++i){ // Loop over output neurons
			for(int j=0; j < R; ++j){
				ls[k+1].x[i] += (ls[k].x[i + j] * ls[k].W[i*R + j]); // MAC
			}
			ls[k+1].x[i] += ls[k].b; // Add bias
		}
		// Apply activation function (sigmoid)
		//#pragma omp parallel for
		for(int i=0; i < ls[k+1].N; ++i)
			ls[k+1].x[i] = activation(ls[k+1].x[i]);

	}
}


// Access to layers, instead of accessing directly to arrays

void kernel(layer_t l, double* y){
// Kernel function: given layer (inputs, weights, bias),
// compute the activations

    #pragma omp parallel for
    // Matrix multiplication
    for(int i=0; i < l.N - R + 1; ++i){ // Loop over output neurons
        y[i] = l.b; // Initialize to bias
        for(int j=0; j < R; ++j){
            y[i] += (l.x[i + j] * l.W[i*R + j]); // MAC
        }
        activation(&y[i]);
    }
    // Free useless memory
    free(l.x);
    free(l.W);
}

// Define propagation function
void forward(layer_t* ls, int K, double* output){
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