#include "include/model.h"
#include "include/ops.h"

void FFN::forward(std::vector<float>& input, std::vector<float>& output){
    // gate = w1 * input
    matmul(gate, input, w1, config.dim, config.hidden_dim);
    
    std::vector<float> gate_silu = silu_copy(gate);
    
    // up = w2 * input
    matmul(up, input, w2, config.dim, config.hidden_dim);
    
    // intermediate = gate_silu * up
    for(int i = 0; i < config.hidden_dim; i++){
        gate[i] = gate_silu[i] * up[i];
    }
    
    // output = w3 * intermediate
    matmul(output, gate, w3, config.hidden_dim, config.dim);
}

int Llama::forward(const int token_id){
    //actually double verify if the pointer already starts from start position
    const float* vec = tok_embeddings + token_id * config.dim; // basically take the pointer, move to the token_id row(* dim size) to advance rows
};