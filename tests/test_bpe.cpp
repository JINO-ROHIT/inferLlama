#include <iostream>
#include "../src/bpe.cpp"

int main(){
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, "model/tokenizer.bin", 256);
    print_tokenizer(&tokenizer);

    printf("%s\n", decode(&tokenizer, 0, 100));

}