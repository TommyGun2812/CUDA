#include <iostream>
#include <thread>
#include <mutex>

using namespace std; 

// needed macros
#define CHAIRS 4
#define CLIENTS 7 
#define ITERATIONS 5

//Mutex synchronize threads
mutex mtx, customermtx, barbermtx, barberDone, customerDone; 

//Critical setion
int client_counter = 0; 

void customer(int id){
    for(int i = 0; i < ITERATIONS; i++){
        mtx.lock();
        if(client_counter == CLIENTS){
            cout << "Client " << id << " goes because there is no space" << "\n"; 
            mtx.unlock(); 
            continue; 
        } 

        client_counter++;
        mtx.unlock();

        customermtx.unlock(); 
        barbermtx.lock();
        cout << "Customer " << id << " asks for a haircut" << "\n"; 
        customerDone.unlock(); 
        barberDone.lock();
        
        mtx.lock(); 
        client_counter--;
        mtx.unlock();
    }
}

void barber(){
    while(1){
        customermtx.lock(); 
        barbermtx.unlock();
        cout << "Barber is cutting the hair" << "\n"; 
        customerDone.lock(); 
        barberDone.unlock();
    }
}

int main(int argc, char* argv[]){
    thread clients[CLIENTS]; 
    mtx.unlock(); 
    customermtx.lock(); 
    barbermtx.lock(); 
    customerDone.lock(); 
    barberDone.lock(); 
    thread barber_thread(barber);

    for(int i = 0; i < CLIENTS; i++){
        clients[i] = thread(customer, i); 
    }

    for(int i = 0; i < CLIENTS; i++){
        clients[i].join(); 
    }

    barber_thread.detach();
    return 0; 
}