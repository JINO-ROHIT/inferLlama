#include <iostream>
#include "../src/bpe.cpp"

int main(){
    Tokenizer tokenizer;
    tokenizer.build_tokenizer("model/tokenizer.json");

    std::string text = "hello world";
    std::vector<int> tokens = tokenizer.encode(text);

    std::cout << "Encoded: ";
    for (int token : tokens) {
        std::cout << token << " ";
    }
    std::cout << "\n";

    std::string decoded = tokenizer.decode(tokens);
    std::cout << "Decoded: " << decoded << "\n";

    // Debug: see actual tokens
    tokenizer.print_tokens(tokens);
}

// g++ -std=c++20 tests/test_bpe.cpp -ljsoncpp -o test && ./test