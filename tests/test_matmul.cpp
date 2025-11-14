#include <iostream>

#include "../src/ops.cpp"
#include "../src/helper.hpp"

int main() {
    std::vector<float> input = {10, 20, 30};
    std::vector<float> weights = {1, 2, 3, 4, 5, 6};
    std::vector<float> output(2, 0.0f);

    {
        Timer t("matmul operation");
        matmul(output, input, weights, 3, 2);
    }

    std::cout << "Output: ";
    for (auto val : output)
        std::cout << val << " ";
    std::cout << std::endl;
}
