#pragma once

#include <vector>

void rmsnorm(std::vector<float> &input, const std::vector<float> &weights, const float &eps);
void silu(std::vector<float> &input); // check if we need const
void softmax(std::vector<float> &input);
void matmul(std::vector<float> &output, const std::vector<float>& input, const std::vector<float> &weights, const int n, const int d);