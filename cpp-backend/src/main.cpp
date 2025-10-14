#include <chrono>
#include <iostream>
#include <string>
#include <thread>

int main() {
    std::cout << "Hello from the C++ backend!" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Ready." << std::endl;
    return 0;
}