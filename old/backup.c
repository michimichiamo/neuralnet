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