#include "../include/bpe.h"
#include <algorithm>
#include <sstream>

void Tokenizer::build_tokenizer(const std::string& json_path) {
    std::ifstream tokenizer_file(json_path, std::ifstream::binary);
    if (!tokenizer_file.is_open()) {
        std::cerr << "cannot open tokenizer\n";
        return;
    }

    Json::Value file;
    tokenizer_file >> file;

    const Json::Value& vocab = file["model"]["vocab"];
    const Json::Value& merges = file["model"]["merges"];

    for (const auto& key : vocab.getMemberNames()) {
        int id = vocab[key].asInt();
        encoder[key] = id;
        decoder[id] = key;
    }

    for(const auto& m : merges){
        merge_pair.push_back(m.asString());
    }

    std::cout << "Loaded vocab size: " << encoder.size() << "\n";
    std::cout << "Loaded merges : " << merge_pair.size() << "\n";
}

std::string Tokenizer::pre_process(std::string input){
    std::string out = "_";
    for(char c: input){
        if(c == ' '){
            out += "_";
        }
        else{
            out += c;
        }
    }
    return out;
}

std::vector<int> Tokenizer::encode(std::string input){
    input = pre_process(input);
    
    std::vector<std::string> tokens;
    for (size_t i = 0; i < input.size(); i++) {
        if ((input[i] & 0xC0) != 0x80) { // Start of a new character
            tokens.push_back(std::string(1, input[i]));
        } else {
            // Continuation byte, append to last token
            if (!tokens.empty()) {
                tokens.back() += input[i];
            }
        }
    }
    
    // Apply BPE merges iteratively
    bool changed;
    do {
        changed = false;
        
        // Try each merge rule in order
        for (const auto& merge_rule : merge_pair) {
            std::istringstream iss(merge_rule);
            std::string first, second;
            if (!(iss >> first >> second)) continue;
            
            std::string combined = first + second;
            
            // Check if combined token exists in vocabulary
            if (encoder.find(combined) == encoder.end()) {
                continue;
            }
            
            // Try to apply this merge
            std::vector<std::string> new_tokens;
            size_t i = 0;
            
            while (i < tokens.size()) {
                if (i < tokens.size() - 1 && 
                    tokens[i] == first && 
                    tokens[i+1] == second) {
                    // Merge found - replace the pair with combined token
                    new_tokens.push_back(combined);
                    i += 2;
                    changed = true;
                } else {
                    // No merge - keep the token as is
                    new_tokens.push_back(tokens[i]);
                    i += 1;
                }
            }
            
            if (changed) {
                tokens = new_tokens;
                break; // Restart from the beginning with new token list
            }
        }
    } while (changed);
    
    std::vector<int> token_ids;
    for (const auto& token : tokens) {
        auto it = encoder.find(token);
        if (it != encoder.end()) {
            token_ids.push_back(it->second);
        } else {
            // fallback: try to handle unknown tokens by splitting into bytes
            std::cout << "Warning: Unknown token '" << token << "', using byte fallback\n";
            for (unsigned char c : token) {
                std::string byte_str(1, static_cast<char>(c));
                auto byte_it = encoder.find(byte_str);
                if (byte_it != encoder.end()) {
                    token_ids.push_back(byte_it->second);
                } else {
                    // ultimate fallback: use unknown token
                    token_ids.push_back(encoder["<unk>"]);
                }
            }
        }
    }
    
    return token_ids;
}

std::string Tokenizer::decode(std::vector<int> token_ids){
    std::string output = "";
    for(int token_id : token_ids){
        auto it = decoder.find(token_id);
        if (it != decoder.end()) {
            std::string token = it->second;
            
            // undo "_" with spaces, but be careful about the first character
            if (!output.empty() && token == "_") {
                output += " ";
            } else {
                output += token;
            }
        } else {
            std::cout << "Warning: Unknown token ID " << token_id << " during decoding\n";
            output += "<?>";
        }
    }
    
    // Remove the leading underscore that was added in pre-processing
    if (!output.empty() && output[0] == '_') {
        // std::cout << output;
        output = output.substr(1);
        // std::cout << output;
    }
    
    // Additional cleanup: replace any remaining underscores with spaces
    // This handles cases where underscores were used as word separators
    std::string final_output;
    for (size_t i = 0; i < output.size(); i++) {
        if (output[i] == '_' && (i == 0 || output[i-1] != '\\')) {
            final_output += ' ';
        } else {
            final_output += output[i];
        }
    }
    
    //std::cout << final_output;
    return final_output;
}

void Tokenizer::print_tokens(const std::vector<int>& token_ids) {
    std::cout << "Tokens [" << token_ids.size() << "]: ";
    for (int id : token_ids) {
        auto it = decoder.find(id);
        if (it != decoder.end()) {
            std::string token = it->second;

            std::string display_token;
            for (char c : token) {
                if (c == ' ') {
                    display_token += "_"; //spaces are replaced by _ for nicer view
                } else {
                    display_token += c;
                }
            }
            std::cout << "'" << display_token << "'(" << id << ") ";
        } else {
            std::cout << "<?>(" << id << ") ";
        }
    }
    std::cout << "\n";
}