#ifndef IALModel_hpp
#define IALModel_hpp

#include <iostream>
#include <cmath>
#include <assert.h>
#include <torch/script.h>

/*
NOTE:
I DISABLED LIBJACK FROM THE PORTAUDIO CMAKE BUILD
BECAUSE MY COMPUTER WONT BUILD AUDACITY WITH IT

I NEED TO GO BACK AND FIX THAT
*/

class IALModel{

    std::vector<std::string> classNames;
    torch::jit::script::Module jitModel;
    torch::jit::script::Module loadModel(const std::string &filepath);

    public:
        // constructor: IALModel model(wxFileName(FileNames::ResourcesDir(), wxT("ial-model.pt")).GetFullPath().ToStdString());
        IALModel(const std::string &filepath);

        const std::vector<std::string> &getClassNames() {return classNames;}

        torch::Tensor downmix(const torch::Tensor audioBatch);
        torch::Tensor reshapeFromBlob(const torch::Tensor audio);

        torch::Tensor predictClassProbabilities(const torch::Tensor audioBatch);
        std::vector<std::string>  predictInstruments(const torch::Tensor audioBatch);

        void modelTest();
        void modelTest(torch::Tensor inputAudio);

};

#endif
