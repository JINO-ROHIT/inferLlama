#pragma once

#include <vector>

std::vector<float> elementWiseMultiply(std::vector<float>& vec1, std::vector<float>& vec2);
void rmsnorm(std::vector<float> &input, const std::vector<float> &weights, const float &eps);
void rmsnorm(std::vector<float> &input, const float* weights, const float &eps);
void silu(std::vector<float> &input); // check if we need const
std::vector<float> silu_copy(const std::vector<float>& input);
void softmax(std::vector<float> &input);
void matmul(std::vector<float> &output, const std::vector<float>& input, const std::vector<float> &weights, const int n, const int d);
void matmul(std::vector<float> &output, const std::vector<float>& input, const float* weights, const int n, const int d);
void matmul_transposed(std::vector<float> &output, const std::vector<float>& input, const float* weights, const int n, const int d);

struct ROPEParams {
    std::vector<float> freqs_cos;
    std::vector<float> freqs_sin;
};

ROPEParams precompute_rope_frequencies(int dim, int max_seq_len, float base=10000.0f);
void apply_rope(std::vector<float>& qk_vec, int dim, int pos, const std::vector<float>& freqs_cos, const std::vector<float>& freqs_sin);