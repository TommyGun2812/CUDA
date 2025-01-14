#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cmath>
#include "utils.h"
using namespace std; 
using namespace std::chrono;

using namespace std;
using namespace std::chrono;

#define SIZE 5000000
#define THREADS std::thread::hardware_concurrency()

typedef struct{
    int start, end; 
    double sumsec;
}Block; 

bool is_prime(int value){
    if(value == 2){
        return true;
        }

    else if(value % 2 == 0){
        return false; 
        }
    else{
        for(int i = 3; i < (sqrt(value)+1); i+=2){
            if(value % i == 0){
                return false; 
            }
        }
    }
    return true; 
}

void sum(Block &b){
    b.sumsec = 0; 
    for(int i = b.start; i < b.end; i++){
        if (is_prime(i) && i > 1){
            b.sumsec += i; 
        }
    }
}

int main(int argc, char* argv[]){
   
    double total_sum; 
    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    int blockSize;
    Block blocks[THREADS];
    thread threads[THREADS];
 

    blockSize = ceil((double) SIZE/ THREADS);

    for (int i = 0; i < THREADS; i++) {
        blocks[i].sumsec = 0; 
        blocks[i].start = (i * blockSize);
        blocks[i].end = (i != (THREADS - 1))? ((i + 1) * blockSize) : SIZE;
    }

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        for (int i = 0; i < THREADS; i++) {
            threads[i] = thread(sum, std::ref(blocks[i]));
        }
        
        total_sum = 0;
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
            total_sum += blocks[i].sumsec;
        }

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
   
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    cout << total_sum << "\n"; 
    
    return 0;
}