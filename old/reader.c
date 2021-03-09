#include <stdio.h>
#include <stdlib.h>

#define R 3


// Define struct to store inputs, weights and bias for each layer
typedef struct layer{
    int N;
    double* x;
    double* W;
    double b;
} layer;

// Fill given array of given size with random values in the range [0,1]
void fill_array(double* array, int n)
{
    for(int i=0; i<n; ++i)
        array[i] = (double) rand() / RAND_MAX;
}
// Read values from given file and store into given structs
void read_values(char* filename, layer* ls, int N, int K){

    // Create file to read values
    FILE* fptr;
    fptr = fopen(filename,"r");
    // Exit with error if file was not correctly read
    if(fptr == NULL)
    {
       printf("An error occurred while reading the file.\n");
       exit(1);           
    }

    printf("Reading from file \"%s\":\n", filename);

    // Read input for first layer from file
    for(int i=0; i<ls[0].N; ++i){
        fscanf(fptr, "%lf\t", &(ls[0].x[i]));
    }
    fscanf(fptr, "\n");
    for(int k=0; k<K; ++k){
//        printf("\tLayer #%d: ", k+1);
        // Set # of weights (R elements for each input neuron)
        int w_N = ls[k].N * R;
//        printf("%d weights\n", w_N);
        // Read weights from file
        for(int i=0; i<w_N; ++i){
            fscanf(fptr,"%lf\t", &(ls[k].W[i]));
        }
        // Read bias from file
        fscanf(fptr, "\n%lf\n", &(ls[k].b));
    }

    // Close file
    fclose(fptr);
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

    // Set name of input file
    char* filename = "values.txt";

    //  Instantiate a struct for each layer to store inputs, weights and bias
    //  (plus one to store the final output)
    layer ls[K+1];

    // Read values from file and store in structs
    read_values(filename, ls, N, K);

    return 0;
}
