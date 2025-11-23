#include "include/sampler.h"

int Sampler::sample_argmax(std::vector<float> probabilities){
    int max_idx = 0;
    float max_prob = probabilities[0];
    for(size_t i = 1; i < probabilities.size(); i++){
        if(probabilities[i] > max_prob){
            max_idx = i;
            max_prob = probabilities[i];
        }
    }

    return max_idx;
}

int Sampler::sample_multi(std::vector<float> probabilities, float coin){
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < probabilities.size(); i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return probabilities.size() - 1; 
}

//TO-DO - read and implement sample top-p
int sample_topp(std::vector<float> probabilities, float topp, ProbIndex probindex, float coin){
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    return 0;
}
