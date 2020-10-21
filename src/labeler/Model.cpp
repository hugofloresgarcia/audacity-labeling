#include "Model.hpp"

AudioClassificationModel::AudioClassificationModel(const std::string &filepath){
    // logger
    classificationLogger.open (wxFileName(FileNames::ResourcesDir(), wxT("labeler-log.txt")).GetFullPath().ToStdString());
    classificationLogger << "loading jit model" << "\n";

    jitModel = loadModel(filepath);

    classificationLogger << "jit model loaded" << "\n";

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

    classificationLogger << "classname set" << "\n";
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
        classificationLogger << e.what() << "\n";
        throw e;
    }
    
    return classifierModel;
}

/*
Downsample audio tensor
the tensor must be shape (batch, channels, time)
*/
torch::Tensor AudioClassificationModel::downmix(const torch::Tensor audioBatch) {
    assert (audioBatch.dim() == 3);

    // take the mean over the channel dimension
    torch::Tensor downmixedAudio =  audioBatch.mean(1, true);
    return downmixedAudio;
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

bool AudioClassificationModel::modelTest(){

    try{
        // prepare input tensor (dummy)
        classificationLogger << "creating input audio" << "\n";
        auto inputAudio = torch::randn({4, 2, 48000});

        classificationLogger << " downmixing audio" << "\n";
        inputAudio = downmix(inputAudio);

        classificationLogger << "doing predictions:" << "\n";
        std::vector<std::string> predictions = predictInstruments(inputAudio);
        
        // log labels
        for (const auto &e : predictions) classificationLogger << e << "\n";
    }
    catch (const std::exception &e){ // hopefully, the exception subclasses the std exception so this would work
        classificationLogger << e.what() << "\n";
    }
    catch (...) {
        classificationLogger << "an unknown error occured" << "\n";
    }

    classificationLogger.close();
}