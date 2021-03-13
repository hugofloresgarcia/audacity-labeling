#ifndef ClassificationModel_h
#define ClassificationModel_h

#include <iostream>
#include <cmath>
#include <assert.h>
#include <torch/script.h>
#include "DeepModel.h"

class ClassificationModel : public DeepModel {
public:
    using DeepModel::DeepModel;
    torch::Tensor predict(const torch::Tensor inputAudio, bool addSoftmax = true);
    std::vector<std::string> predictFromAudioFrame(const torch::Tensor audioBatch, float confidenceThreshold);
    std::vector<std::string> predictFromAudioSequence(const torch::Tensor audioSequence, float confidenceThreshold);
    std::vector<std::string> constructLabelsFromProbits(const torch::Tensor confidences, const torch::Tensor indices, 
                                            float confidenceThreshold);

    void modelTest(torch::Tensor inputAudio);
};


#endif
