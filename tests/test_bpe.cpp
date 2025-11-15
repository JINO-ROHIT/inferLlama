#include <iostream>
#include "../src/bpe.cpp"

int main(){
    Tokenizer tokenizer;
    tokenizer.build_tokenizer("model/tokenizer.json");
    tokenizer.print_top_n(10);

    std::cout << tokenizer.decode(std::vector<int> {10994, 11526});
}

