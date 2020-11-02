#include "IALModel.hpp"


// NOTE: would be better to specify filepaths for both 
// the compiled model AND the instruments. 
IALModel::IALModel(const std::string &filepath){
    jitModel = loadModel(filepath);

    // hardcoding these here for now
    classNames  = {
        // these are the medleydb classes
        "accordion","acoustic guitar","alto saxophone","auxiliary percussion","bamboo flute","banjo","baritone saxophone","bass clarinet","bass drum","bassoon","bongo","brass section","cello","clarinet","clean electric guitar","distorted electric guitar","dizi","double bass","drum machine","drum set","electric bass","electric piano","erhu","female vocalist","flute","french horn","glockenspiel","gong","gu","guzheng","harmonica","harp","horn section","lap steel guitar","male vocalist","mandolin","melodica","oboe","other","oud","percussion","piano","piccolo","soprano saxophone","string section","synthesizer","tabla","tenor saxophone","timpani","trombone","trumpet","tuba","vibraphone","viola","violin","vocalists","yangqin","zhongruan",
        // these are the philharmonia classes
    //    "saxophone", "flute" , "guitar", "contrabassoon",
    //    "bass-clarinet","trombone","cello","oboe",
    //    "bassoon", "banjo", "mandolin", "tuba", "viola",
    //    "french-horn", "english-horn", "violin", "double-bass",
    //    "trumpet", "clarinet"
    };
    // sort classnames
    std::sort(classNames.begin(), classNames.end());
}

/*
load a classifier model
*/
torch::jit::script::Module IALModel::loadModel(const std::string &filepath) {
    torch::jit::script::Module classifierModel;
    try {
        classifierModel = torch::jit::load(filepath);
        classifierModel.eval();
    }
    catch (const c10::Error& e) {
        std::cerr << "Error Loading Model" << std::endl;
        std::cout << e.what() << "\n";
        throw e;
    }

    return classifierModel;
}

/*
downmix audio tensor
input tensor must be shape (batch, channels, time)
*/
torch::Tensor IALModel::downmix(const torch::Tensor audioBatch) {
    assert (audioBatch.dim() == 3);
    // if channel dim is already 1, return
    if (audioBatch.sizes()[1] == 1 ){
        return audioBatch;
    }

    // take the mean over the channel dimension
    torch::Tensor downmixedAudio =  audioBatch.mean(1, true);
    return downmixedAudio;
}

/*
reshape from blob to size (batch, channels, time)
with each batch having shape (1, 48000)
audio that comes in needs to be MONO already and should be shape 
(samples,)
*/
torch::Tensor IALModel::reshapeFromBlob(const torch::Tensor audio){

    auto length = audio.sizes()[0];
    int newLength = length - length % 48000;
    
    // debug: log the first 200 samples
    // for (int i = 0; i < 200; i++) std::cout << audio.index(torch::indexing::TensorIndex(i)) << " ";
    
    torch::Tensor reshapedAudio = audio.index({torch::indexing::Slice(0, newLength, 1)});
    reshapedAudio = reshapedAudio.reshape({-1, 1, 48000});
    return reshapedAudio;
}

/*
returns probits with shape (batch, n_classes)
*/
torch::Tensor IALModel::predictClassProbabilities(const torch::Tensor audioBatch){
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(audioBatch);
    
    // get class probabilities
    auto probits = jitModel.forward({inputs}).toTensor();
    std::cout << probits << '\n';
    return probits;
}

std::vector<std::string> IALModel::predictInstruments(const torch::Tensor audioBatch){
    auto probits = predictClassProbabilities(audioBatch);
    auto [confidences, indices] = probits.max(1, false);

    std::vector<std::string> predictions;
    // iterate through confs and idxs
    for (int i = 0; i < confidences.sizes()[0]; ++i){
        // grab our confidence mesaure
        auto conf = confidences.index({i}).item().to<float>();
        auto idx = indices.index({i}).item().to<int>();

        // return a not-sure if the probability is less than 0.3
        std::string prediction;
        if (conf < 0.3){
            prediction = "not-sure";
        } else {
            prediction = classNames.at(int(idx));
        }
        
        predictions.push_back(prediction);
    }

    return predictions;
}

void IALModel::modelTest(torch::Tensor inputAudio){

    try{
        std::cout << " downmixing audio" << "\n";
        inputAudio = downmix(inputAudio);

        std::cout << "doing predictions:" << "\n";
        std::vector<std::string> predictions = predictInstruments(inputAudio);
        
        // log labels
        for (const auto &e : predictions) std::cout << e << "\n";
    }
    catch (const std::exception &e){ // hopefully, the exception subclasses the std exception so this would work
        std::cout << e.what() << "\n";
    }
    catch (...) {
        std::cout << "an unknown error occured" << "\n";
    }
}
