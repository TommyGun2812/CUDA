#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <atomic>
#include <random>
#include <map>
#include <vector>
#include <set>

class Barbershop {
private:
    static const int NUM_CHAIRS = 4;
    static const int NUM_CLIENTS = 7;
    static const int ATTEMPTS_PER_CLIENT = 5;

    std::mutex mutex;
    std::condition_variable barberCV;
    std::condition_variable customerCV;
    std::queue<int> waitingRoom;
    bool barberSleeping;
    std::atomic<bool> shopOpen;

    // Estadísticas por cliente
    struct ClientStats {
        int successfulHaircuts = 0;
        int failedAttempts = 0;
        int totalAttempts = 0;
        bool isBeingServed = false;  // Nuevo: indica si el cliente está siendo atendido
    };
    std::map<int, ClientStats> clientStats;

    // Set para clientes en sala de espera
    std::set<int> waitingClients;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<> waitDist;

public:
    Barbershop() : 
        barberSleeping(true), 
        shopOpen(true),
        gen(rd()),
        waitDist(1000, 3000) {
        // Inicializar estadísticas para cada cliente
        for (int i = 1; i <= NUM_CLIENTS; ++i) {
            clientStats[i] = ClientStats();
        }
    }

    void barber() {
        while (shopOpen || !waitingRoom.empty()) {
            std::unique_lock<std::mutex> lock(mutex);

            if (waitingRoom.empty()) {
                std::cout << "Barbero se va a dormir\n";
                barberSleeping = true;
                barberCV.wait(lock, [this] { 
                    return !waitingRoom.empty() || !shopOpen; 
                });
                barberSleeping = false;
                if (!shopOpen && waitingRoom.empty()) break;
            }

            int clientId = waitingRoom.front();
            waitingRoom.pop();
            waitingClients.erase(clientId);  // Cliente sale de la sala de espera
            clientStats[clientId].isBeingServed = true;  // Marcar que está siendo atendido

            std::cout << "Barbero comienza a cortar el pelo al cliente " << clientId << "\n";
            lock.unlock();

            std::this_thread::sleep_for(std::chrono::milliseconds(2000));

            lock.lock();
            std::cout << "Barbero termino de cortar el pelo al cliente " << clientId << "\n";
            clientStats[clientId].successfulHaircuts++;
            clientStats[clientId].isBeingServed = false;  // Ya no está siendo atendido
            customerCV.notify_all();
        }
        std::cout << "Barbero cierra la tienda\n";
    }

    void customer(int id) {
        while (clientStats[id].totalAttempts < ATTEMPTS_PER_CLIENT && shopOpen) {
            std::unique_lock<std::mutex> lock(mutex);

            // Verificar si el cliente ya está en la sala de espera o siendo atendido
            if (waitingClients.count(id) > 0 || clientStats[id].isBeingServed) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(waitDist(gen)));
                continue;
            }

            clientStats[id].totalAttempts++;

            if (waitingRoom.size() < NUM_CHAIRS) {
                waitingRoom.push(id);
                waitingClients.insert(id);  // Marcar que está en la sala de espera
                std::cout << "Cliente " << id << " entra a la sala de espera. (" 
                          << clientStats[id].totalAttempts << "/5 intentos) Sillas: " 
                          << waitingRoom.size() << "/" << NUM_CHAIRS << "\n";
                
                if (barberSleeping) {
                    std::cout << "Cliente " << id << " despierta al barbero\n";
                    barberCV.notify_one();
                }
            } else {
                std::cout << "Cliente " << id << " se va, no hay sillas disponibles. (" 
                          << clientStats[id].totalAttempts << "/5 intentos)\n";
                clientStats[id].failedAttempts++;
                lock.unlock();
                
                std::this_thread::sleep_for(std::chrono::milliseconds(waitDist(gen)));
                continue;
            }

            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(waitDist(gen)));
        }
    }

    void startDay() {
        std::vector<std::thread> clients;
        std::thread barberThread(&Barbershop::barber, this);

        for (int i = 0; i < NUM_CLIENTS; ++i) {
            clients.emplace_back(&Barbershop::customer, this, i + 1);
        }

        for (auto& client : clients) {
            client.join();
        }

        shopOpen = false;
        barberCV.notify_one();
        barberThread.join();

        // Mostrar estadísticas detalladas
        std::cout << "\nEstadisticas del dia:\n";
        for (const auto& [clientId, stats] : clientStats) {
            std::cout << "\nCliente " << clientId << ":\n";
            std::cout << "  - Intentos totales: " << stats.totalAttempts << "/5\n";
            std::cout << "  - Cortes exitosos: " << stats.successfulHaircuts << "\n";
            std::cout << "  - Intentos fallidos: " << stats.failedAttempts << "\n";
        }
    }
};

int main() {
    std::cout << "Abriendo la barberia...\n\n";
    Barbershop shop;
    shop.startDay();
    return 0;
}