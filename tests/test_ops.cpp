#include <iostream>
#include <vector>

#include "../src/ops.cpp" // import headers in actual files and compile src
#include "../src/helper.hpp"   // for Timer

void test_matmul() {

    std::vector<float> input = {10, 20, 30};
    std::vector<float> weights = {1, 2, 3, 4, 5, 6};
    std::vector<float> output(2, 0.0f);

    {
        Timer t("matmul");
        matmul(output, input, weights, 3, 2);
    }

    std::cout << "Output: ";
    for (auto val : output)
        std::cout << val << " ";
    std::cout << "\n";
}

void test_rmsnorm() {

    std::vector<float> input = {1.0f, 2.0f, 10.0f};
    std::vector<float> weights = {1.0f, 1.0f, 1.0f};
    float eps = 1e-6f;

    {
        Timer t("rms norm");
        rmsnorm(input, weights, eps);
    }

    std::cout << "Output: ";
    for (auto v : input) std::cout << v << " ";
    std::cout << "\n";
}

void test_softmax() {

    std::vector<float> input = {1.0f, 2.0f, 10.0f};
    {
        Timer t("softmax");
        softmax(input);
    }

    std::cout << "Output: ";
    for (auto v : input) std::cout << v << " ";
    std::cout << "\n";
}

int main() {
    test_matmul();
    test_rmsnorm();
    test_softmax();
    return 0;
}
