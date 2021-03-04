#include <stdio.h>
#include <stdlib.h>

#define R 3

// Fill given array of given size with random values in the range [0,1]
void fill_array(double* array, int n)
{
    for(int i=0; i<n; ++i)
        array[i] = (double) rand() / RAND_MAX;
}

void write_values(char* filename, int N, int K){

    // Create file to store values
    FILE* fptr;
    fptr = fopen(filename,"w");
    
    // Return error if file was not created
    if(fptr == NULL)
    {
       printf("An error occurred while creating the file.\n");
       exit(1);           
    }

    printf("Writing to file \"%s\"\n", filename);



    // Fill input array with random values
    double* x = (double*)malloc(N*sizeof(double));
    fill_array(x, N);

    // Write input array to file
    for(int i=0; i<N; ++i){
        fprintf(fptr, "%lf\t", x[i]);
    }
    fprintf(fptr, "\n");

    for(int k=0; k<K; ++k){
        printf("Iteration #%d: ", k+1);
        printf("%d inputs, ", N);
        // Set # of weights (R elements for each input neuron)
        int w_N = N * R;
        printf("%d weights\n", w_N);
        // Fill weights
        double* W = (double*)malloc(w_N *sizeof(double));
        fill_array(W, w_N);
        // Fill bias
        double b = (double) rand()/RAND_MAX;
        // Write weights and bias to file
        for(int i=0; i<w_N; ++i){
            fprintf(fptr,"%lf\t", W[i]);
        }
        fprintf(fptr, "\n%lf\n", b);
        // Set # of input neurons
        N = N - R + 1;
    }

    // Close file
    fclose(fptr);
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
    	printf("The network consists of %d layers. Layer 0 has size %d.\n", K,N);

    }
    else {
    	printf("Please provide the dimension of the first layer (N) and the number of layers (K)");
    	return 1;
    }

    // Set the seed
    srand(42);

    // Set name of output file
    char* filename = "values.txt";

    // Generate values and write to file
    write_values(filename, N, K);

    return EXIT_SUCCESS;
}
