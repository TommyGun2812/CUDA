#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cmath>
#include "utils.h"
using namespace std; 
using namespace std::chrono;

#define SIZE    1000000000 // 1e9
#define THREADS std::thread::hardware_concurrency()

typedef struct{
    int *a, s, start, end; 
} Block; 

void evennumbers(Block &b){
    int sum = 0;
    for(int i = b.start; i < b.end; i++){
        if (b.a[i] % 2 == 0){
            sum += 1; 
        }
    } 
    b.s = sum; 
}

int main(int argc, char* argv[]){
    //Variable declaration
    int *a, sum; 
    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    int blockSize;
    Block blocks[THREADS];
    thread threads[THREADS];

    a = new int[SIZE]; 
    fill_array(a, SIZE);
    

    blockSize = ceil((double) SIZE/ THREADS);

    for (int i = 0; i < THREADS; i++) {
        blocks[i].a = a;
        blocks[i].s = 0; 
        blocks[i].start = (i * blockSize);
        blocks[i].end = (i != (THREADS - 1))? ((i + 1) * blockSize) : SIZE;
    }

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        for (int i = 0; i < THREADS; i++) {
            threads[i] = thread(evennumbers, std::ref(blocks[i]));
        }
        sum=0;
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
            sum += blocks[i].s;
        }

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    display_array("a:", a);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    cout << sum << "\n";
    delete [] a;

    return 0;
}
