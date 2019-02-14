#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <curand_kernel.h>
#include <unistd.h>
#include <assert.h>

const int dims = 16;
const int results_capacity = 10000;

__device__ int results_index = 0;
__device__ bool not_done = true;

__global__ void zero_results(float (*results)[dims + 1]) {

    for (int i = 0; i < results_capacity; i++) {
    	for (int j = 0; j < dims + 1; j++) {
	    results[i][j] = 0;
	}
    }
}

__global__ void my_kernel(float (*results)[dims + 1]) {

    int id = blockIdx.x*blockDim.x+threadIdx.x;

    curandState_t state;
    curand_init(0, id, 0, &state);

    // printf("%d\n", id);

    while (results_index < results_capacity) {

	float v[dims];
	for (int i = 0; i < dims; i++) {
	    v[i] = curand_uniform(&state);
	}

	double length = 0;
	for (int i = 0; i < dims; i++) {
	    length += v[i]*v[i];
	}
	length = sqrt(length);

	if (length < 1) {

	    int ind = atomicAdd(&results_index, 1);
	    if (ind < results_capacity) {
		for (int i = 0; i < dims; i++) {
		    results[ind][i] = v[i];
		}
		results[ind][dims] = length;
	    } else {
	    	printf("Averted race condition.\n");
	    }
	}
    }
}
 
int main() {

    cudaError_t rv;

    rv = cudaDeviceReset();
    assert(rv == cudaSuccess);

    float (*d_results)[dims + 1];
    rv = cudaMalloc(&d_results, results_capacity*sizeof(*d_results));
    assert(rv == cudaSuccess);

    /*
    cudaMalloc(&d_ip, 4);
    int i = 0;
    cudaMemcpy(d_ip, &i, 4, cudaMemcpyHostToDevice);
    */

    printf("Kernel start.\n");
    my_kernel<<<15, 128>>>(d_results);
    assert(cudaPeekAtLastError() == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    printf("Kernel stop.\n");

    float (*results)[dims + 1];
    results = (float (*)[dims + 1]) malloc(results_capacity*sizeof(*d_results));

    rv = cudaMemcpy(results, d_results, results_capacity*sizeof(*d_results), cudaMemcpyDeviceToHost);
    assert(rv == cudaSuccess);
    rv = cudaFree(d_results);
    assert(rv == cudaSuccess);

    for (int i = 0; i < results_capacity; i++) {
	float length = 0;
	for (int j = 0; j < dims; j++) {
		length += results[i][j]*results[i][j];
	}
	length = sqrt(length);
	assert(fabs(length - results[i][dims]) < 1E-5);
    }

    /*
    for (int i = 0; i < results_capacity; i++) {
    	printf("length(");
	for (int j = 0; j < dims; j++) {
	    printf("%f, ", results[i][j]);
	}
	printf(") = %f\n", results[i][dims]);
    }
    */
}
