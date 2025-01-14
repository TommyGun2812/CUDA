#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <algorithm>
#include "utils.h"

using namespace std; 
using namespace std::chrono; 

#define SIZE 1000000
#define THREADS 512
#define BLOCKS min(32, ((SIZE/THREADS)+1))

__global__ void EulerPi(double *results){
    __shared__ double cache[THREADS];
    unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    double piSum = 0.0; 

    while(idx < SIZE){
        if(idx > 0){
            double number = idx;
            piSum += 1.0 / (number * number);
        }

        idx += (blockDim.x * gridDim.x);
    }

    cache[threadIdx.x] = piSum; 
    __syncthreads();

    int gap = blockDim.x / 2; 
    while(gap > 0){
        if(threadIdx.x < gap){
            cache[threadIdx.x] += cache[threadIdx.x + gap];
        }
        __syncthreads(); 
        gap /= 2; 
    }

    if(threadIdx.x == 0){
        results[blockIdx.x] = cache[0];
    }    
}

int main(){
    double *results, *device_results, pi; 

    high_resolution_clock::time_point start, end; 
    double timeElapsed;

    results = new double[BLOCKS];
    cudaMalloc((void**) &device_results, BLOCKS * sizeof(double));
    
    cout << "Starting...\n";
    for(int j = 0; j < N; j++){
        start = high_resolution_clock::now();

        EulerPi <<< BLOCKS, THREADS >>>(device_results);

        end = high_resolution_clock::now();
        timeElapsed += duration<double, std::milli>(end - start).count();
    }
    
    cudaMemcpy(results, device_results, BLOCKS * sizeof(double),cudaMemcpyDeviceToHost); 

    pi = 0.0;
    for(int i = 0; i < BLOCKS; i++){
        pi += results[i];
    }

    pi = sqrt(6.0 * pi);
    cout << "Pi: " << fixed << setprecision(20) << pi <<"\n";
    cout << "avg time = " << fixed << setprecision(3) << (timeElapsed / N) <<  " ms\n";

    delete[] results;
    cudaFree(device_results);
    return 0;
}