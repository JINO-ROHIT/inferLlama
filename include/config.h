#pragma once

#include <unordered_map>

struct Config {
    int dim = 4096;                           // hidden_size
    int hidden_dim = 11008;                   // intermediate_size  
    int n_layers = 32;
    int n_heads = 32;
    int n_kv_heads = 32;
    int vocab_size = 32000;
    int max_seq_len = 4096;                   // max_position_embeddings
    float rope_theta = 10000.0f;
    float norm_eps = 1e-5f;
};