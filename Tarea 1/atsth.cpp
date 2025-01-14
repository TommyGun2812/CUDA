#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cmath>
#include "utils.h"
using namespace std; 
using namespace std::chrono;

#define SIZE    1000000000 // 1e9

int evennumbers(int *a,int size){
    int sum = 0;
     for(int i = 0; i < size; i++){
        if (a[i] % 2 == 0){
            sum++; 
        }
    }
    return sum;
}

int main(int argc, char* argv[]){
    //Variable declaration
    int *a, sum; 
    a = new int [SIZE]; 
    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    sum = 0; 
    fill_array(a, SIZE);

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        evennumbers(a, SIZE);

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }

    display_array("a:", a);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";
    
    sum = evennumbers(a, SIZE);
    cout << sum << "\n"; 
    delete [] a;
    return 0; 
}