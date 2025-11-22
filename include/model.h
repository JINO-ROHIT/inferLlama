#pragma once

#include <vector>

#include "config.h"
#include "loader.h"
#include "ops.h"


// rms norm layer (i think we need it since it has weights)
class RMSNorm{
    private:
        const float* weight;
        float eps = 1e-5f; // dont hardcode find a way to load this from a common config instead of passing each time(also for all classes)
    
    public:
        RMSNorm(): weight(nullptr){}

        void load_weights(const Loader& loader, const std::string& layer_name){
            const auto& tensor = loader.get_tensor(layer_name);
            weight = tensor.data<float>();
        }

        //forward?
};

class FFN{
    private:
        Config config;

        const float* w1;
        const float* w2; 
        const float* w3; //ptrs to store the weights

        size_t w1_size, w2_size, w3_size;
    
        // activation buffers
        std::vector<float> gate, up, down;
    public:
        FFN(const Config& cfg): config(cfg), w1(nullptr), w2(nullptr), w3(nullptr){
            gate.resize(config.hidden_dim);
            up.resize(config.hidden_dim);
            down.resize(config.dim);
        }

        void load_weights(const Loader& loader, int layer_idx){
            std::string prefix = "model.layers." + std::to_string(layer_idx) + ".mlp.";
        
            const auto& gate_tensor = loader.get_tensor(prefix + "gate.proj.weight");
            const auto& up_tensor = loader.get_tensor(prefix + "up.proj.weight");
            const auto& down_tensor = loader.get_tensor(prefix + "down.proj.weight");

            w1 = gate_tensor.data<float>();
            w2 = down_tensor.data<float>();
            w3 = up_tensor.data<float>();
            
            w1_size = gate_tensor.size();
            w2_size = down_tensor.size();
            w3_size = up_tensor.size();
            
            std::cout << "loaded FFN weights for layer " << layer_idx << std::endl;
        }

        void forward(std::vector<float>& input, std::vector<float>& output);
};


class Llama{
    private:
        Config config;

        const float* tok_embeddings;
        std::vector<FFN> ffn_layers;
    public:
        Llama(const Config& cfg): config(cfg){
            for(int i = 0; i < config.n_layers; ++i){
                ffn_layers.emplace_back(FFN(config));
            }
        }

        void build(Loader& loader){
            loader.load_weights("model/bin_files");

            //load the embedding layer
            const auto& embed_tensor = loader.get_tensor("model.embed.tokens.weight");
            tok_embeddings = embed_tensor.data<float>();  //pointer to the float array

            std::cout << embed_tensor.size(); // [32000, 4096]
            // FFN ffn_layer(config);
            // ffn_layer.load_weights(loader, 0);
        }

        //TO-DO
        int forward(const int token_id);
};