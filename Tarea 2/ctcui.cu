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
#define SIZE 10000 // 5e6
#define THREADS 512
#define BLOCKS min(32, ((SIZE/THREADS)+1))

__global__ void rank_sort(int* array, int* sorted_array){
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
    int counter = 0; 

    while(idx < SIZE){
        counter = 0; 
        for(int i = 0; i < SIZE; i++){
             if((array[idx] > array[i]) || (array[idx] == array[i] && idx < i)){
                counter++; 
            }  
        }
        sorted_array[counter] = array[idx];
        idx += (blockDim.x * gridDim.x);
    }
}

int main(){
    int *array, *sorted_array, *device_array, *device_sortedArray;

    high_resolution_clock::time_point start, end; 
    double timeElapsed; 

    array = new int [SIZE]; 
    random_array(array, SIZE);

    sorted_array = new int [SIZE]; 
    display_array("array", array);

    cudaMalloc((void**) &device_array, SIZE * sizeof(int));
    cudaMalloc((void**) &device_sortedArray, SIZE * sizeof(int));

    cudaMemcpy(device_array, array, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cout << "Starting...\n"; 
    for (int j = 0; j < N; j++){
        start = high_resolution_clock::now(); 
         
         rank_sort <<< BLOCKS, THREADS >>>(device_array, device_sortedArray);

         end = high_resolution_clock::now();
         timeElapsed += duration<double, std::milli>(end - start).count(); 
    }
     cudaMemcpy(sorted_array, device_sortedArray, SIZE * sizeof(int),cudaMemcpyDeviceToHost);
 

     for(int i = 0; i < 100; i++){
        cout << sorted_array[i] << " " << "\n";
     }

     cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    delete [] array;
    delete [] sorted_array; 
    cudaFree(device_array); 
    cudaFree(device_sortedArray);

    return 0;
}

