#include <iostream>
using namespace std;

int main(){
    int counter = 0; 
    int prev ; 

    for(int i = 0; i <= 5; i++){
        prev = ++counter; 
        cout << "prev: " << prev << " " << "current: " << counter << "\n"; 
    }
}
