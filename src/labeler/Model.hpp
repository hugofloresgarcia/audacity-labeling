#ifndef Model_hpp
#define Model_hpp

#include <iostream>
#include <cmath>
#include <assert.h>
#include <torch/script.h>


class AudioClassificationModel{

    std::vector<std::string> classNames;
    torch::jit::script::Module jitModel;
    torch::jit::script::Module loadModel(const std::string &filepath);

    public:
        AudioClassificationModel(const std::string &filepath);

        const std::vector<std::string> &getClassNames() { return classNames; }

        torch::Tensor downsample(const torch::Tensor audioBatch);

        torch::Tensor predictClassProbabilities(const torch::Tensor audioBatch);
        std::vector<std::string>  predictInstruments(const torch::Tensor audioBatch);
};

#endif