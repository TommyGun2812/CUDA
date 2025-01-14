#include <iostream>
#include <cmath>
#include <iomanip>
#include <chrono>
#include "utils.h"

using namespace std; 
using namespace std::chrono; 

#define NUMBER 1000000
#define N 10

double EulerPi(int x){
    double pi = 0; 
    double sum = 0; 

    for(int i = 1; i <= x; i++){
        double number = i;
        sum += 1.0 / (number * number);

    }

    sum *= 6; 
    pi = sqrt(sum);

    return pi;
}

int main(){

    double pi = EulerPi(NUMBER);
     // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;
    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        // We take a clock record before execution.
        start = high_resolution_clock::now();

        // We perform the task.
        EulerPi(NUMBER);

        // We take a clock record after execution. We calculate the 
        // difference between the two records. This difference is 
        // the time it took to execute the task.
        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    // We display the result and the average execution time.
    cout << "Pi = " << fixed << setprecision(20) << pi << "\n";
    cout << "avg time = " << fixed << setprecision(3) << (timeElapsed / N) <<  " ms\n";
    
    return 0;
}