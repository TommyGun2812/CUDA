#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cmath>
#include "utils.h"
using namespace std; 
using namespace std::chrono;

#define SIZE    10000// 1e4
#define THREADS std::thread::hardware_concurrency()

typedef struct{
    int *a, start, end; 
}Block; 

void rankSort(Block &b,int* sa, int size){
    int counter; 
    for(int i = b.start; i < b.end; i++){
        counter = 0; 
        for(int j = 0; j < size; j++){
            if(i != j && b.a[i] > b.a[j]){
                counter++; 
            } else if(b.a[i] == b.a[j] && (j > i)){
                counter++;
            }
        }
       sa[counter]=b.a[i];
    }
}

int main(int argc, char* argv[]){
    int *a, *sa; 

    a = new int [SIZE]; 
    sa = new int [SIZE]; 

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    int blockSize;
    Block blocks[THREADS];
    thread threads[THREADS];

    random_array(a, SIZE); 

    blockSize = ceil((double) SIZE/ THREADS);

    for (int i = 0; i < THREADS; i++) {
        blocks[i].a = a; 
        blocks[i].start = (i * blockSize);
        blocks[i].end = (i != (THREADS - 1))? ((i + 1) * blockSize) : SIZE;
    }

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        for (int i = 0; i < THREADS; i++) {
            threads[i] = thread(rankSort, std::ref(blocks[i]), sa, SIZE);
        }
        
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
        }

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    display_array("a:", a);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";
     
    delete [] a;
    delete [] sa; 

    return 0;
}