#pragma once

#include <vector>

std::vector<float> elementWiseMultiply(std::vector<float>& vec1, std::vector<float>& vec2);
void rmsnorm(std::vector<float> &input, const std::vector<float> &weights, const float &eps);
void silu(std::vector<float> &input); // check if we need const
std::vector<float> silu_copy(const std::vector<float>& input);
void softmax(std::vector<float> &input);
void matmul(std::vector<float> &output, const std::vector<float>& input, const std::vector<float> &weights, const int n, const int d);
void matmul(std::vector<float> &output, const std::vector<float>& input, const float* weights, const int n, const int d);