#ifndef IALModel_hpp
#define IALModel_hpp

#include <iostream>
#include <cmath>
#include <assert.h>
#include <torch/script.h>

class IALModel{

    std::vector<std::string> instruments;
    torch::jit::script::Module jitModel;
    torch::jit::script::Module loadModel(const std::string &filepath);
    std::vector<std::string> loadInstrumentList(const std::string &filepath);

    const int chunkLen = 48000;

    public:
        // constructor: IALModel model(wxFileName(FileNames::ResourcesDir(), wxT("ial-model.pt")).GetFullPath().ToStdString());
        IALModel(const std::string &filepath);

        const std::vector<std::string> &getClassNames() {return classNames;}
        const int getChunkLen() {return chunkLen;}

        torch::Tensor downmix(const torch::Tensor audioBatch);
        torch::Tensor padAndReshape(const torch::Tensor audio);

        torch::Tensor predictClassProbabilities(const torch::Tensor audioBatch);
        std::vector<std::string>  predictInstruments(const torch::Tensor audioBatch, float confidenceThreshold);

        void modelTest(torch::Tensor inputAudio);
};

#endif
