#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>

#include <json/json.h>

class Tokenizer {
private:
    std::map<std::string, int> encoder;
    std::map<int, std::string> decoder;
    std::vector<std::pair<std::string, std::string>> merges; // not storing atm

public:
    void build_tokenizer(const std::string& json_path);
    std::string decode(std::vector<int> token_ids);
    void print_top_n(int N) const; // pass decoder or have to sort encoder
};
