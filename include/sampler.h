#pragma once

#include <vector>

typedef struct{
    float prob;
    int index;
} ProbIndex;

class Sampler{
    private:
        int vocab_size;
        ProbIndex* probindex;
        float temperature;
        float topp;
        unsigned long long rng_state;

    public:
        int sample_argmax(std::vector<float> probabilities);
        int sample_multi(std::vector<float> probabilities, float coin);
        int sample_topp(std::vector<float> probabilities, float topp, ProbIndex probindex, float coin);
};