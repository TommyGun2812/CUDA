#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <thread>
#include <cmath>

using namespace std;
using namespace std::chrono;

#define SIZE        10000
#define Random      10000
#define N 			10
#define limit 	    10
#define THREADS     thread::hardware_concurrency()


typedef struct{
    int start,end,*arr;
} Block;

void display_array(int *array, int size) {
    cout<<"[ ";
    for (int i = 0; i < size; i++){
        cout<<array[i];
        if (i != (size-1)){cout<<",";}
    }
    cout<<" ]"<<endl;
}

void fill_array(int *array, int size) {
   for (int i = 0; i < size; i++) {
      array[i] = rand() % (Random + 1);
   }
}

void task1(Block &block, int *arrF) {
    int rank;
    for (int i = block.start; i < block.end; i++){
        rank = 0;
        for (int j = 0; j < SIZE; j++){
            if (i != j && block.arr[i] > block.arr[j]){
                rank++;
            }
            else if (block.arr[i] == block.arr[j] && (j > i)){
                rank++;
            }
        }
        arrF[rank]=block.arr[i];
    }
}

int main (int argc, char* argv[]){
   int *array, *arrayf;

   high_resolution_clock::time_point start, end;

   array  = new int[SIZE];
   arrayf = new int[SIZE];
   double timeElapsed;

   fill_array(array,SIZE);

   int blockSize = ceil(SIZE / (double) THREADS);
   thread threads[THREADS];
   Block blocks[THREADS];

   for (int i = 0; i < THREADS; i++){
        blocks[i].start = i * blockSize;
        blocks[i].arr = array;
        if (((i +1) * blockSize)>SIZE){blocks[i].end = SIZE;}
        else {blocks[i].end = (i +1) * blockSize;}
    }

   cout << "Starting...\n";
   timeElapsed = 0;
   for (int j = 0; j < N; j++) {
      start = high_resolution_clock::now();

      for (int i = 0; i < THREADS; i++){
        threads[i] = thread(task1,ref(blocks[i]), arrayf);
      }

      for (int i = 0; i < THREADS; i++){
        threads[i].join();
      }

      end = high_resolution_clock::now();
      timeElapsed += 
      duration<double, std::milli>(end - start).count();
   }
   cout << "Before = ";
   display_array(array, limit);
   cout << "After = ";
   display_array(arrayf, limit);
   cout << "avg time = " << fixed << setprecision(3) << (timeElapsed / N) <<  " ms\n";

   return 0;

   delete[] array;
   delete[] arrayf;
}