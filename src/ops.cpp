#include <cmath>
#include <vector>

void silu(std::vector<float>& input){
    for(int i = 0; i < input.size(); i ++){
        input[i] = input[i] * ( 1.0 / 1.0 + exp(-input[i]));
    }
}

std::vector<float> silu_copy(const std::vector<float>& input) {
    std::vector<float> result = input; // make a copy
    for(int i = 0; i < result.size(); i++) {
        result[i] = result[i] * (1.0f / (1.0f + exp(-result[i])));
    }
    return result;
}

void rmsnorm(std::vector<float> &input, const std::vector<float> &weights, const float &eps){
    // 1. take the square of the values
    // 2. take the mean of the squares
    // 3. rms = take the sqrt(mean + small eps)
    // 4. divided input values / rms
    // 5. element wise multiply learnable weights

    float ss = 0.0f;
    int n = input.size();

    for(int i = 0; i < n; i++){
        ss += input[i] * input[i];
    }

    float mean = ss / n;
    float rms = sqrt(mean + eps);

    for(int i = 0; i < n; i++){
        input[i] = (input[i] / rms)  * weights[i];
    }

}

void softmax(std::vector<float> &input){
    float max_val = 0.0f;
    int n = input.size();

    for(int i = 0; i < n; i++){
        if (input[i] > max_val){
            max_val = input[i];
        } 
    }

    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        input[i] = expf(input[i] - max_val);
        sum += input[i];
    }
    // normalize
    for (int i = 0; i < n; i++) {
        input[i] /= sum;
    }
    
}

void matmul(std::vector<float> &output, const std::vector<float>& input, const std::vector<float> &weights, const int n, const int d){
    // matrix-vector multiplication: output = weights * input
    // weights: [d x n]
    // input:   [n]
    // output:  [d]
    for(int i = 0; i < d; i++){
        float val = 0.0f;
        for(int j = 0; j < n; j ++){
            val += weights[i * n + j] * input[j];
        }
        output[i] = val;
    }

}

//overload with pointer as input for weights
void matmul(std::vector<float> &output, const std::vector<float>& input, 
                const float* weights, const int n, const int d){
    for(int i = 0; i < d; i++){
        float val = 0.0f;
        for(int j = 0; j < n; j++){
            val += weights[i * n + j] * input[j];
        }
        output[i] = val;
    }
}