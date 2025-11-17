#include <fstream>
#include <iostream>
#include <vector>
#include <filesystem>

#define SAFETENSORS_IMPLEMENTATION
#include "../include/safetensors.hpp"
#include "../include/model.h"


std::vector<char> read_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Can't open " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::uintmax_t file_size = std::filesystem::file_size(filename);
    
    std::vector<char> buffer(file_size);
    if (!file.read(buffer.data(), file_size)) {
        std::cerr << "Can't read file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    
    return buffer;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <safetensors_file>" << std::endl;
        return 1;
    }
    
    std::string filename = argv[1];
    auto file_buffer = read_file(filename);
    
    safetensors_File f = {0};
    char* result = safetensors_file_init(file_buffer.data(), file_buffer.size(), &f);

    int embed_index = safetensors_lookup(&f, "model.embed_tokens.weight");
    if (embed_index == -1) {
        std::cout << "Embedding tensor not found!" << std::endl;
        return 1;
    }
    
    safetensors_TensorDescriptor& embed_tensor = f.tensors[embed_index];
    
    std::cout << "Embedding Tensor Info:" << std::endl;
    std::cout << "  Vocabulary size: " << embed_tensor.shape[0] << " tokens" << std::endl;
    std::cout << "  Embedding dimension: " << embed_tensor.shape[1] << std::endl;
    std::cout << "  Data type: " << safetensors_dtype_name(embed_tensor.dtype) << std::endl;
    std::cout << std::endl;
}