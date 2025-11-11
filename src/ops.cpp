#include <cmath>
#include <vector>

void rmsnorm(std::vector<float> &input, std::vector<float> &weights, float &eps){
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