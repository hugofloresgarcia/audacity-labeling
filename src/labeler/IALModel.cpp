#include "IALModel.hpp"


// NOTE: would be better to specify filepaths for both 
// the compiled model AND the instruments. 
/*
@brief: creates a classifier instance
@params:
    std::string &modelPath: path to .pt file (must be a torch::jit model)
*/
IALModel::IALModel(const std::string &modelPath){
    jitModel = loadModel(modelPath);

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
@brief: returns a torch::jit::model specified by @param std::string filepath
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
@brief: pads an audio tensor with shape (samples,)
     with the necessary zeros and then reshapes to (batch, 1, sampleRate)
@params:
    torch::Tensor audio: MONO audio tensor with shape (samples,)
@returns:
    torch::Tensor reshapedAudio: audio tensor with shape (batch, 1, sampleRate)
*/
torch::Tensor IALModel::padAndReshape(const torch::Tensor audio){
    auto length = audio.sizes()[0];
    
    // RIGHT: pad with zeros to meet length
    int padLength = ceil(length / sampleRate) * sampleRate - length;
    torch::Tensor padTensor = torch::zeros({padLength});
    torch::Tensor reshapedAudio = torch::cat({audio, padTensor})

    reshapedAudio = reshapedAudio.reshape({-1, 1, sampleRate});
    return reshapedAudio;
}

/* 
TODO: the already-compiled models return log-probabilities. 
this should be fixed on the python side though, to keep this class architecture agnostic. 
@brief: forward pass through the model and get raw class probabilities as output
@params:
    torch::Tensor audioBatch: batch of mono audio with shape (batch, 1, sampleRate)
@returns:
    torch::Tensor probits: per-class probabilities with shape (batch, n_classes)
*/
torch::Tensor IALModel::predictClassProbabilities(const torch::Tensor audioBatch){
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(audioBatch);
    
    // get class probabilities
    auto probits = jitModel.forward({inputs}).toTensor();
    std::cout << probits << '\n';
    return probits;
}

/*
@briefs: forward pass through the model and get a list of classes with the highest probabilities for every instance in the batch. 
@params: 
    torch::Tensor audioBatch: batch of mono audio with shape (batch, 1, sampleRate)
    float confidenceThreshold: probabilities under this value will be labeled 'not-sure'
@returns:  
    std::vector<std::string> predictions: list of class predictions for every instance in the batch
*/
std::vector<std::string> IALModel::predictInstruments(const torch::Tensor audioBatch, float confidenceThreshold = 0.3){
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
        if (conf < confidenceThreshold){
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
