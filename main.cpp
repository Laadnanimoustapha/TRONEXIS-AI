#include <iostream>

int main() {
        std::cout << "Please enter your text: ";
    std::string user_text;
    std::getline(std::cin, user_text);
    std::cout << "You entered: " << user_text << std::endl;
    return 0;
}