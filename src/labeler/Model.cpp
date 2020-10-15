#include "Model.hpp"

AudioClassificationModel::AudioClassificationModel(const std::string &filepath){
    jitModel = loadModel(filepath);
    // hardcoding these here for now
    classNames  = {
        "saxophone", "flute" , "guitar", "contrabassoon",
        "bass-clarinet","trombone","cello","oboe",
        "bassoon", "banjo", "mandolin", "tuba", "viola",
        "french-horn", "english-horn", "violin", "double-bass",
        "trumpet", "clarinet"
    };
    // sort classnames
    std::sort(classNames.begin(), classNames.end());
}   

/*
load a classifier model
*/
torch::jit::script::Module AudioClassificationModel::loadModel(const std::string &filepath) {
    torch::jit::script::Module classifierModel;
    try {
        classifierModel = torch::jit::load(filepath);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error Loading Model" << std::endl;
        throw e;
    }
    
    return classifierModel;
}

/*
Downsample audio tensor
the tensor must be shape (batch, channels, time)
*/
torch::Tensor AudioClassificationModel::downsample(const torch::Tensor audioBatch) {
    assert (audioBatch.dim() == 3);
    // take the mean over the channel dimension
    torch::Tensor downmixedAudio =  audioBatch.mean(1, true);
}

/*
returns probits with shape (batch, n_classes)
*/
torch::Tensor AudioClassificationModel::predictClassProbabilities(const torch::Tensor audioBatch){
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(audioBatch);

    // get class probabilities
    auto probits = jitModel.forward({inputs}).toTensor();

    return probits;
}

std::vector<std::string> AudioClassificationModel::predictInstruments(const torch::Tensor audioBatch){
    auto probits = predictClassProbabilities(audioBatch);
    auto [confidences, indices] = probits.max(1, false);

    std::vector<std::string> predictions;
    // iterate through confs and idxs
    for (int i = 0; i < confidences.sizes()[0]; ++i){
        // grab our confidence mesaure
        auto conf = confidences.index({i}).item().to<float>();
        auto idx = indices.index({i}).item().to<int>();

        // return a not-sure if the probability is less than 0.5
        std::string prediction;
        if (conf < 0.5){
            prediction = "not-sure";
        } else {
            prediction = classNames.at(int(idx));
        }
        
        predictions.push_back(prediction);
    }

    return predictions;
}