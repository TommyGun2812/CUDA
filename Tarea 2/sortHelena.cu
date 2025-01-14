#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cuda_runtime.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 	10000
#define THREADS 512
#define BLOCKS min(32, ((SIZE / THREADS) + 1))

__global__ void ranksort(int *array, int *result){
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    int rank = 0;

    while(tid < SIZE){
        rank = 0;
        for (int i = 0; i < SIZE; i++) {
            if (array[tid] > array[i] || (array[tid] == array[i] && tid < i)) {
                rank++;
            }
        }
        result[rank] = array[tid];
        tid += (blockDim.x * gridDim.x);
    }
}

int main() {
    int *array, *sorted;
    int *deviceA, *deviceS;

    high_resolution_clock::time_point start, end;
    double timeElapsed;

    array = new int[SIZE];
    sorted = new int[SIZE];

    random_array(array, SIZE);
    display_array("Array:", array);

    cudaMalloc( (void**) &deviceA, SIZE * sizeof(int) );
    cudaMalloc( (void**) &deviceS, SIZE * sizeof(int) );

    cudaMemcpy(deviceA, array, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();
        cout << "coca1"<<endl;
        ranksort<<<BLOCKS, THREADS>>> (deviceA, deviceS);
        cout << "coca2"<<endl;
        end = high_resolution_clock::now();
        timeElapsed += duration<double, std::milli>(end - start).count();
    }

    cudaMemcpy(sorted, deviceS, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    display_array("Arreglo ordenado: ", sorted);

    cout << "Tiempo en prueba paralela con cuda: " << fixed << setprecision(3)<< (timeElapsed / N) <<  " ms\n";

    cudaFree(deviceA);
    cudaFree(deviceS);

    delete [] array;
    delete [] sorted;

    return 0;
}