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
    sums = 0; 
    for(int i = start; i <= end; i++){
        if(i == 0){
            continue;
        }
        double number = i; 
        sums += 1.0  / (number * number);
    }
}

int main(){
    double pi; 
    cout << "Threads = " << THREADS << "\n";

    int blockSize = ceil ((double) SIZE / THREADS);
    thread threads[THREADS];
    double partialSums[THREADS];

    for(int i = 0; i < THREADS; i++){
        int start = i * blockSize; 
        int end = (i != (THREADS - 1))? ((i + 1) * blockSize) : SIZE;
        threads[i] = thread(EulerPi, start, end, std::ref(partialSums[i]));
    }

    pi = 0;
    for(int i = 0; i < THREADS; i++){
        threads[i].join();
        pi += partialSums[i];
    }

    pi = sqrt(6.0 * pi);

    cout << "Pi = " << pi << "\n";
}