#include <iostream>
#include <iomanip>
#include <chrono>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 5000000

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

double sum(int size){
    double sum = 0; 
    for(int i = 0; i <= size; i++){
        if (is_prime(i) && i > 1){
            sum += i; 
        }
    }
    return sum; 
}


int main() {
    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    long final_sum = sum(SIZE); 

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        sum(SIZE);

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }

    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";
    
    cout << final_sum << "\n"; 
    return 0; 

}

