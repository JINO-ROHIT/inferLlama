#include "../include/bpe.h"

//like strip in python
// "   hello world  " becomes "hello world"
static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    size_t end = s.find_last_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

void Tokenizer::build_tokenizer(const std::string& json_path) {
    std::ifstream tokenizer_file(json_path, std::ifstream::binary);
    if (!tokenizer_file.is_open()) {
        std::cerr << "cannot open tokenizer\n";
        return;
    }

    Json::Value file;
    tokenizer_file >> file;

    const Json::Value& vocab = file["model"]["vocab"];
    for (const auto& key : vocab.getMemberNames()) {
        int id = vocab[key].asInt();
        encoder[key] = id;
        decoder[id] = key;
    }
    std::cout << "Loaded vocab size: " << encoder.size() << "\n";
}

//TO-DO: implement proper bpe with fallbacks, this is quite dum dum
std::string Tokenizer::decode(std::vector<int> token_ids){
    std::string output = "";
    for(int token_id: token_ids){
        output += decoder[token_id];
    }
    return output;
}

void Tokenizer::print_top_n(int N) const {
    int count = 0;
    for (const auto& [k, v] : decoder) {
        std::cout << "[" << k << "] = " << v << "; ";
        if (++count >= N) break;
    }
    std::cout << "\n";
}