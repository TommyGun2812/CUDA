#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cmath>
#include "utils.h"
#include <cuda_runtime.h>
#include <algorithm>

using namespace std; 
using namespace std::chrono; 

//Define constants
#define SIZE 5000001 // 5e6
#define THREADS 512
#define BLOCKS min(32, ((SIZE/THREADS)+1))

__device__ bool is_prime(int value){
    if(value == 2){
        return true;
        }

    else if(value % 2 == 0){
        return false; 
        }
    else{
        for(int i = 3; i <= (sqrtf(static_cast<float>(value))); i+=2){
            if(value % i == 0){
                return false; 
            }
        }
    }
    return true; 
}

__global__ void sum(double *results){
    __shared__ double cache[THREADS]; 
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
    double prime_sum = 0.0; 

    while(idx < SIZE){
        if(is_prime(idx)){
            prime_sum += idx; 
        }
        idx += (blockDim.x * gridDim.x); 
    }

    cache[threadIdx.x] = prime_sum; 
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


int main(int argc, char* argv[]){
    double *results, *device_results, result; 

    high_resolution_clock::time_point start, end; 
    double timeElapsed; 

    results = new double[BLOCKS]; 

    cudaMalloc((void**) &device_results, BLOCKS * sizeof(double));

    cout << "Starting...\n"; 
    for (int j = 0; j < N; j++){
        start = high_resolution_clock::now(); 
         
         sum <<< BLOCKS, THREADS >>>(device_results);

         end = high_resolution_clock::now();
         timeElapsed += duration<double, std::milli>(end - start).count(); 
    }
     cudaMemcpy(results, device_results, BLOCKS * sizeof(double),cudaMemcpyDeviceToHost);

     result = 0.0; 

     for(int i = 0; i < BLOCKS; i++){
        result += results[i];
     }

     cout << "Prime numbers: " << fixed << setprecision(0) << result << "\n"; 
     cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    delete [] results;

    cudaFree(device_results);
    return 0;

}