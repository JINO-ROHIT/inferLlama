#include <iostream>
#include <chrono>

#include "include/loader.h"
#include "include/model.h"
#include "include/bpe.h"
#include "include/sampler.h"

// TO-DO - handle for EOS token 
std::string generate(std::string prompt, int steps){

    Tokenizer tokenizer;
    Loader loader;
    Config config;
    Sampler sampler;

    Llama llama(config);

    std::vector<int> input_ids;

    tokenizer.build_tokenizer("model/tokenizer.json");
    llama.build(loader); // load model weights


    std::vector<int> tokens = tokenizer.encode(prompt);

    for (int token : tokens) {
        input_ids.emplace_back(token);
    }

    std::string empty_prompt = "";
    if(prompt.empty()){
        prompt = empty_prompt;
    };

    int token = input_ids[0];
    int pos = 0;
    int next; // next token

    while(pos < steps){
        std::vector<float> logits = llama.forward(token);
        prompt += tokenizer.decode({token});
        pos += 1;
        next = sampler.sample_argmax(logits);
        token = next;

    }
    return prompt;
}

int main() {
    auto start =  std::chrono::high_resolution_clock::now();
    std::string prompt = "hey, ";
    int generation_steps = 10;

    std::string output = generate(prompt, generation_steps);
    std::cout << "\nGenerated Text: \n" << output << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\nTotal generation time: " << duration.count() << " ms" << std::endl;
}