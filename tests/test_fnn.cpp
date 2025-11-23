#include <chrono>

#include "include/config.h"
#include "include/model.h"
int main() {
    Config config;

    Loader loader;

    Llama llama{config}; 
    llama.build(loader); // temp inconvenience for for a better tomorrow

    FFN ffn(config);
    ffn.load_weights(loader, 0);
    std::vector<float> input(config.dim, 1.0f);
    std::vector<float> output(config.dim);
    
    // for warmup
    ffn.forward(input, output);
    
    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 100;
    
    for (int i = 0; i < iterations; i++) {
        ffn.forward(input, output);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Average FFN forward time: " 
              << duration.count() / iterations << " microseconds" << std::endl;
}