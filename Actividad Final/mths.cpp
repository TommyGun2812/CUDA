#include <iostream>
#include <cmath>
#include <thread>
#include <iomanip>
#include <chrono>
#include "utils.h"

using namespace std; 
using namespace std::chrono; 

#define SIZE 1000000
#define THREADS std::thread::hardware_concurrency()
#define N 10

void EulerPi(int start, int end, double& sums){
    double local_sums = 0.0; 

    for(int i = start; i <= end; i++){
        if(i == 0){
            continue;
        }
        double number = i; 
        local_sums += 1.0 / (number * number);
    }

    sums = local_sums;
}

int main(){
    double pi; 

    high_resolution_clock::time_point start, end; 
    double timeElapsed; 

    cout << "Threads = " << THREADS << "\n";

    int blockSize = ceil((double) SIZE / THREADS);
    thread threads[THREADS];
    double partialSums[THREADS];

    cout << "Starting... \n";
    timeElapsed = 0; 
    
    for(int j = 0; j < N; j++){
         start = high_resolution_clock::now(); 

         for(int i = 0; i < THREADS; i++){
            int rangeStart = i * blockSize; 
            int rangeEnd = (i != (THREADS - 1)) ? ((i + 1) * blockSize) : SIZE; 
            threads[i] = thread(EulerPi, rangeStart, rangeEnd, std::ref(partialSums[i]));
         }

         pi = 0.0;
         for(int i = 0; i < THREADS; i++){
            threads[i].join();
            pi += partialSums[i];
         }

         end = high_resolution_clock::now();
         timeElapsed += duration<double, std::milli>(end - start).count();
    }

    pi = sqrt(6.0 * pi);
    cout << "Pi = " << fixed << setprecision(20) << pi << "\n";
    cout << "avg time = " << fixed << setprecision(3) << (timeElapsed / N) << " ms/n" << "\n";

    return 0;
}