#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>

#include <json/json.h>

class Tokenizer {
private:
    std::unordered_map<std::string, int> encoder; // unordered map is faster
    std::unordered_map<int, std::string> decoder;
    std::vector<std::string> merge_pair;

public:
    void build_tokenizer(const std::string& json_path);
    std::vector<int> encode(std::string input);
    std::string decode(std::vector<int> token_ids);
    void print_tokens(const std::vector<int>& token_ids); 
    
private:
    std::string pre_process(std::string input);
};
