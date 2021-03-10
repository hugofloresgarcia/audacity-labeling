#ifndef IALModel_hpp
#define IALModel_hpp

#include <iostream>
#include <cmath>
#include <assert.h>
#include <torch/script.h>

class IALModel {

    std::vector<std::string> instruments;
    torch::jit::script::Module jitModel;
    torch::jit::script::Module loadModel(const std::string &filepath);
    std::vector<std::string> loadInstrumentList(const std::string &filepath);

    const int chunkLen = 48000;

    public:
        IALModel();
        // constructor: IALModel model(wxFileName(FileNames::ResourcesDir(), wxT("ial-model.pt")).GetFullPath().ToStdString());
        IALModel(const std::string &modelPath, const std::string &instrumentListPath);

        const std::vector<std::string> &getInstrumentList() {return instruments;}
        const int getChunkLen() {return chunkLen;}

        torch::Tensor downmix(const torch::Tensor audioBatch);
        torch::Tensor padAndReshape(const torch::Tensor audio);

        torch::Tensor modelForward(const torch::Tensor inputAudio, bool addSoftmax);
        std::vector<std::string> predictFromAudioFrame(const torch::Tensor audioBatch, float confidenceThreshold);
        std::vector<std::string> predictFromAudioSequence(const torch::Tensor audioSequence, float confidenceThreshold);
        std::vector<std::string> constructLabelsFromProbits(const torch::Tensor confidences, const torch::Tensor indices, 
                                                            float confidenceThreshold);

        

        void modelTest(torch::Tensor inputAudio);
};

#endif
