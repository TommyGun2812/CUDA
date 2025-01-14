#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>

#define THREADS 4

using namespace std; 
using namespace std::chrono;

int counter = 0; 

void increment(int id){
    for(int i = 0; i < 20; i++){
        cout << "increment " << id << ", previuos counter = " << counter << "\n"; 
        counter++; 
        cout << "increment " << id << ", current counter = " << counter << "\n";
        this_thread::sleep_for(chrono::seconds(1));
    }
}

void decrement(int id){
    for(int i = 0; i < 20; i++){
        cout << "increment " << id << ", previuos counter = " << counter << "\n"; 
        counter--; 
        cout << "decrement " << id << ", current counter = " << counter << "\n";
        this_thread::sleep_for(chrono::seconds(1));
    }
}

int main(int argc, char* argv[]){
    thread my_threads[THREADS]; 

    for(int i = 0; i < THREADS; i++){
        if (i % 2 == 0){
            my_threads[i] = thread(increment, i); 
        } else {
            my_threads[i] = thread(decrement, i); 
        }
    }

    for(int i = 0; i < THREADS; i++){
        my_threads[i].join();
    }

    return 0; 
}