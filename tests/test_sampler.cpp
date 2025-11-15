#include <iostream>
#include <vector>
#include <random>
#include "../src/sampler.cpp"

int main() {
    Sampler sampler;

    std::vector<float> probs = {0.1f, 0.3f, 0.6f};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float coin = dist(gen);

    int idx1 = sampler.sample_argmax(probs);
    int idx2 = sampler.sample_multi(probs, coin);

    std::cout << "Argmax sample: " << idx1 << std::endl;
    std::cout << "Multinomial sample: " << idx2 << std::endl;

    return 0;
}
