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
#define SIZE 1000000000 // 1e9
#define THREADS 512
#define BLOCKS min(32, ((SIZE/THREADS)+1))

__global__ void evenumbers(int *array, int *results){
    __shared__ int cache[THREADS]; 
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int count = 0;

    while(idx < SIZE){
        if(array[idx] % 2 == 0){
            count++;
        }
        idx += (blockDim.x * gridDim.x); 
    }

    cache[threadIdx.x] = count; 
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
    int *array, *results, result; 
    int *device_array, *device_results; 

    high_resolution_clock::time_point start, end; 
    double timeElapsed; 

    array = new int [SIZE]; 
    fill_array(array, SIZE);
    display_array("array", array);

    results = new int[BLOCKS]; 

    cudaMalloc((void**) &device_array, SIZE * sizeof(int));
    cudaMalloc((void**) &device_results, BLOCKS * sizeof(int));

    cudaMemcpy(device_array, array, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cout << "Starting...\n"; 
    for (int j = 0; j < N; j++){
        start = high_resolution_clock::now(); 
         
         evenumbers <<< BLOCKS, THREADS >>>(device_array, device_results);

         end = high_resolution_clock::now();
         timeElapsed += duration<double, std::milli>(end - start).count(); 
    }
     cudaMemcpy(results, device_results, BLOCKS * sizeof(int),cudaMemcpyDeviceToHost);

     result = 0; 

     for(int i = 0; i < BLOCKS; i++){
        result += results[i];
     }

     cout << "Even numbers: " << result << "\n"; 
     cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    delete [] array;

    return 0;

}