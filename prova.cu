#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define R 3
#define BLKDIM 32
/*
__global__ void kernel(int* a, int n){
    const int bx = blockIdx.x;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int id = bx*blockDim.y*blockDim.x + ty*blockDim.x + tx;

//    printf("%d\t%d\t%d\n", bx, ty, tx);

    if(id < n){
        a[id] = id;
    }
}

int check(int* a, int n){
    int id;
    for(id = 0; id < n; ++id){
        if(a[id]!=id){
            printf("Error at index %d: %d\n", id, a[id]);
            return 1;
        }
    }
    return 0;
}

__global__ void myk(int* a, int n, int m){
    int i;
    for(i=0; i<n; ++i)
        a[i] = m*n + i;
}
*/


// Print array to screen
void print_array(float* a, int n){

    printf("Values\n\n");
    for(int i=0; i<n; ++i){
        printf("%f\t", a[i]);
    }
    printf("\n");
}

// Fill given array
void fill_array(float* array, int n){

    for(int i=0; i<n; ++i)
        array[i] = i;
}

typedef struct {
   int width;
   int height;
   float* elements;
} Matrix;

//__device__ float elems[10] = {0,3,6,9,12,15,18,21,24,27};
/*
__device__ void check(Matrix m, float* x, int N){
    int control=0;
    for(int i=0; i<N; ++i){
        if(m.elements[i]!=x[i]){
            printf("Error at index #%d: %f!=%f\n", i, m.elements[i], x[i]);
            control=1;
        }
    }
    if(!control) printf("Check ok.\n");
}
*/
__global__ void kernel(Matrix m, int N){
    printf("Matrix m: (%d, %d)\n", m.width, m.height);
//    printf("Values inside\n\n");
//    for(int i=0; i<N; ++i){
//        printf("%f\t", m.elements[i]);
//    }
//    printf("\n");
//
//    printf("Now let's update them.\n");
    m.width = 3; m.height = 3;
    for(int i=0; i<N; ++i){
        m.elements[i] = i*3;
    }
//    check(m, elems, N);

    }

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

int main(int argc, char* argv[]){

    int N = 10000000;
    int K = 2;
    printf("N:%d\n", N);

    Matrix m[K];
    float* elements[K];
    for (int i = 0; i<2; ++i){
        m[i].width = N;
        m[i].height = N;
        elements[i] = (float*)malloc(N*(i+1)*sizeof(float));
    
        cudaSafeCall(cudaMalloc((void **)&(m[i].elements), N*(i+1)*sizeof(float)));
        fill_array(elements[i], N*(i+1));
        print_array(elements[i], 20*(i+1));
        cudaSafeCall(cudaMemcpy(m[i].elements, elements[i], N*(i+1)*sizeof(float), cudaMemcpyHostToDevice));

    }
    printf("After allocation.\n");
    checkMemory();

    for(int i=0; i<K; ++i){
        kernel<<<1,1>>>(m[i], N*(i+1));
        cudaCheckError();
        cudaSafeCall(cudaMemcpy(elements[i], m[i].elements, N*(i+1)*sizeof(float), cudaMemcpyDeviceToHost));
    }
    printf("After kernel.\n");
    checkMemory();

    print_array(elements[0], 20);
    print_array(elements[1], 40);

    for(int i=0; i<K; ++i){
        cudaFree(m[i].elements);
    }

    printf("After freeing.\n");
    checkMemory();

/*
    int n=10, m=3;
    int a[m][n], d_a[m][n];
    for(int i=0; i<m; ++i)
        for(int j=0; j<n; ++j)
            a[i][j] = 0;

    cudaSafeCall(cudaMalloc((void **)&(d_a), n*m*sizeof(int)));
//    cudaSafeCall(cudaMemcpy(d_a, a, n*m*sizeof(int), cudaMemcpyHostToDevice));
//    dim3 grid((N+BLKDIM*BLKDIM-1)/(BLKDIM*BLKDIM));
//    dim3 block(BLKDIM, BLKDIM);
//    printf("tx\tbx\n");
//    kernel<<<grid, block>>>(d_a, N);

    for(int i=0; i<m; ++i){
        myk<<<1,1>>>(d_a[i], n, i);
    }

    cudaCheckError();
    cudaSafeCall(cudaMemcpy(a, d_a, n*m*sizeof(int), cudaMemcpyDeviceToHost));

//    if(!check(a, N)) printf("Check ok.\n");

*/


/*
    float* h_data, *h_out, *d_data, *d_out;
    h_data = (float*) malloc(3*sizeof(float));
    h_out = (float*) malloc(3*sizeof(float));
    cudaSafeCall(cudaMalloc((void **)&d_data, 5*sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&d_out, 5*sizeof(float)));
    cudaSafeCall(cudaMemset(d_data, 0, 5*sizeof(float)));
    cudaSafeCall(cudaMemset(d_out, 0, 5*sizeof(float)));
    
    for(int i=0; i<3; ++i){
        kernel<<<1,i+1>>>(d_data);
        cudaSafeCall(cudaMemcpy(d_out+2, d_data, (i+1)*sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    cudaSafeCall(cudaMemcpy(h_data, d_data, 3*sizeof(float), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(h_out, d_out, 5*sizeof(float), cudaMemcpyDeviceToHost));
    
    for(int i=0; i<5; ++i){
        printf("%f\t", h_data[i]);
        printf("%f\n", h_out[i]);
    }
    printf("\n");
*/



/*
    float* d_data; float* h_data;
    int R = 10, C = 15;
    int N = R*C;
    int size = N * sizeof(float);

    h_data = (float*)malloc(size);

    for(int i=0; i<C; ++i){
        for(int j=0; j<R; ++j){
            h_data[i*R + j]= (float)(i*R + j);
            printf("%f\t", h_data[i*R + j]);
      }
    }
    printf("\n");
    // allocate device memory
    cudaSafeCall(cudaMalloc((void **)&(d_data), size));
    // copy host array to device                                    
    cudaSafeCall(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    
    for(int i=0; i<3; ++i){
        printf("%f\t", h_data[i]);
    }
    printf("\n");

    kernel<<<1,N>>>(d_data);
    cudaDeviceSynchronize();

    cudaSafeCall(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    for(int i=0; i<3; ++i){
        printf("%f\t", h_data[i]);
    }
    printf("\n");
*/


}