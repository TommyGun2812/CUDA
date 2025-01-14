#include <iostream>
#include <iomanip>
#include <thread>

using namespace std;

#define THREADS     4  //Declaro 4 hilos
#define ITERATIONS  5 // Declaro 5 iteraciones

int counter = 0; // Variable contador global

void increment(int id) {
    int prev;

    for (int i = 0; i < ITERATIONS; i++) {
        prev = counter++;
        cout << "id=" << id << ", previous = " << prev << " current = " << counter << "\n";
    }
}

int main(int argc, char* argv[]) {

    thread threads[THREADS];
    
    for (int i = 0; i < THREADS; i++) {
        threads[i] = thread(increment, i);
    }

    for (int i = 0; i < THREADS; i++) {
        threads[i].join();
    }

    return 0;
}