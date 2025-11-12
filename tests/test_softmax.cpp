#include <iostream>
#include "../src/ops.cpp"

int main() {
    std::vector<float> input = {1.0f, 2.0f, 10.0f};

    softmax(input);

    for (auto v : input) std::cout << v << " ";
    std::cout << std::endl;
}
