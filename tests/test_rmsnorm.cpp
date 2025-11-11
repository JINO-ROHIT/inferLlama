#include <iostream>
#include "../src/ops.cpp"

int main() {
    std::vector<float> input = {1.0f, 2.0f, 10.0f};
    std::vector<float> weights = {1.0f, 1.0f, 1.0f};
    float eps = 1e-6f;

    rmsnorm(input, weights, eps);

    for (auto v : input) std::cout << v << " ";
    std::cout << std::endl;
}
