#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cmath>
#include "utils.h"
using namespace std; 
using namespace std::chrono;

#define SIZE 10000 

void rank_sort(int* array, int size){
    int counter = 0; 
    int ranks_array[size];
    int sorted_array[size]; 

    for(int i = 0; i < size; i++){
        counter = 0; 
        for(int j = 0; j < size; j++){
            if((array[i] > array[j]) || (array[i] == array[j] && i < j)){
                counter++; 
            }
        }
        ranks_array[i] = counter; 
    }

    for(int i = 0; i < size; i++){
        sorted_array[ranks_array[i]] = array[i]; 
    } 
}


int main(int argc, char* argv[]){
    int *array;
    array = new int[SIZE]; 
    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;
    //FIlling the array with random values
    random_array(array, SIZE); 


    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        rank_sort(array, SIZE);

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }

    display_array("array:", array);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";
    
    delete [] array;
    return 0; 

}
