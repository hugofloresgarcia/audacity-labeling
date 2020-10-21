#ifndef Model_hpp
#define Model_hpp

#include <iostream>
#include <cmath>
#include <assert.h>
#include <torch/script.h>

/*
NOTE:
I DISABLED LIBJACK FROM THE PORTAUDIO CMAKE BUILD
BECAUSE MY COMPUTER WONT BUILD AUDACITY WITH IT

I NEED TO GO BACK AND FIX THAT

branch: jack audio
copyClip.Resample(16000);
c
*/

class AudioClassificationModel{

    std::vector<std::string> classNames;
    torch::jit::script::Module jitModel;
    torch::jit::script::Module loadModel(const std::string &filepath);

    std::ofstream classificationLogger; // need to come up with a better way to log.

    public:
        // constructor: AudioClassificationModel model(wxFileName(FileNames::ResourcesDir(), wxT("tunedopenl3_philharmonia_torchscript.pt")).GetFullPath().ToStdString());
        AudioClassificationModel(const std::string &filepath);

        const std::vector<std::string> &getClassNames() {return classNames;}

        torch::Tensor fromBuffer()

        torch::Tensor downmix(const torch::Tensor audioBatch);

        torch::Tensor predictClassProbabilities(const torch::Tensor audioBatch);
        std::vector<std::string>  predictInstruments(const torch::Tensor audioBatch);

        bool modelTest();

};

#endif