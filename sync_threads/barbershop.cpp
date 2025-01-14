#include <iostream>
#include <thread>
#include <mutex>

using namespace std;

// Configuración del problema
#define CHAIRS 4    // Número de sillas disponibles
#define CLIENTS 7   // Número total de clientes
#define ITERATIONS 1 // Solo una iteración para simplificar

// Sección crítica
int client_counter = 0; // Número de clientes esperando

// Mutex para sincronizar el acceso a la variable compartida
mutex clients_lock, barber_lock;

void customer(int id) {
    for (int i = 0; i < ITERATIONS; i++) {
        clients_lock.lock(); // Bloquea para acceder a la sección crítica
        if (client_counter == CHAIRS) {
            // Si no hay sillas disponibles, el cliente se va
            cout << "Cliente " << id << " se va porque no hay sillas disponibles.\n";
            clients_lock.unlock(); // Libera el acceso a la sección crítica
            continue;
        } else {
            // El cliente ocupa una silla
            client_counter++;
            cout << "Cliente " << id << " se sienta. Clientes esperando: " << client_counter << "\n";
            barber_lock.unlock(); // Despierta al barbero si está durmiendo
        }
    
        clients_lock.unlock(); // Libera el acceso a la sección crítica
    }
}

void barber() {
    while (true) {
        barber_lock.lock(); // Espera hasta que un cliente llegue
        clients_lock.lock();

        if (client_counter > 0) {
            // El barbero atiende a un cliente
            cout << "El barbero está atendiendo a un cliente.\n";
            client_counter--; // Disminuye el número de clientes
            cout << "Clientes restantes: " << client_counter << "\n";
        } else {
            // Si no hay clientes, el barbero "duerme" esperando que llegue uno
            cout << "El barbero está durmiendo porque no hay clientes.\n";
            // Se "despierta" cuando un cliente lo desbloquea
        }

        clients_lock.unlock(); // Permite que otro cliente acceda a la sección crítica
    }
}

int main() {
    thread barber_thread(barber);  // Un solo barbero
    thread client_threads[CLIENTS]; // Arreglo de hilos para los clientes

    // Crear los hilos para los clientes
    for (int i = 0; i < CLIENTS; i++) {
        client_threads[i] = thread(customer, i + 1);
    }

    // Iniciar los hilos de los clientes
    for (int i = 0; i < CLIENTS; i++) {
        client_threads[i].join();
    }

    // Terminar el hilo del barbero
    barber_thread.join();

    return 0;
}