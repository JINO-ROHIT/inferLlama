#pragma once

#include <vector>

void rmsnorm(std::vector<float> &input, std::vector<float> &weights, float &eps);
void softmax(std::vector<float> &input);
void matmul(std::vector<float> &output, std::vector<float> input, std::vector<float> &weights, int n, int d);