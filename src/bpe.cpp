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
    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::cerr << "cannot open tokenizer\n";
        return;
    }

    std::string line;
    bool in_model = false;
    bool in_vocab = false;

    while (std::getline(file, line)) {
        line = trim(line);

        if (line.find("\"model\"") != std::string::npos) {
            in_model = true;
            continue;
        }

        if (in_model && line.find("\"vocab\"") != std::string::npos) {
            in_vocab = true;
            continue;
        }

        // exit vocab block
        if (in_vocab && (line == "}," || line == "}")) {
            in_vocab = false;
            continue;
        }

        if (in_vocab) {
            if (line == "{") continue;

            if (!line.empty() && line.back() == ',')
                line.pop_back();

            size_t colon = line.find(':');
            if (colon == std::string::npos) continue;

            std::string key = trim(line.substr(0, colon));
            std::string val = trim(line.substr(colon + 1));

            // remove quotes from token
            if (key.front() == '"' && key.back() == '"')
                key = key.substr(1, key.size() - 2);

            if (val.empty() || !isdigit(val[0]))
                continue;

            int id = std::stoi(val);

            encoder[key] = id;
            decoder[id] = key;
        }
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