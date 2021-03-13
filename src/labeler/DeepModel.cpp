#include "DeepModel.h"

/**
 @brief: creates a classifier instance
 @param modelPath path to jit model (.pt) file.
 @param classlistPath path to class file.
 NOTE: the class file must have each class name separated by a newline
*/
DeepModel::DeepModel(const std::string &modelPath, const std::string &classlistPath){
   jitModel = loadModel(modelPath);
   classes = loadClasslist(classlistPath);
}

/**
 @brief: returns a torch::jit::model specified by filepath
 @param filepath a string containing the filepath
*/
torch::jit::script::Module DeepModel::loadModel(const std::string &filepath) {
   torch::jit::script::Module model;
   try {
      model = torch::jit::load(filepath);
      model.eval();
   }
   catch (const c10::Error& e) {
      std::cerr << "Error Loading Model" << std::endl;
      std::cout << e.what() << "\n";
      throw e;
   }

   return model;
}

/**
 @brief: returns a torch::jit::model specified by filepath
 @param filepath location of torch model
*/
std::vector<std::string> DeepModel::loadClasslist(const std::string &filepath) {
   std::vector<std::string> classes;
   std::ifstream classFile(filepath);

   if (classFile.is_open()) {
      while (classFile.good()) {
         std::string line;
         getline(classFile, line);
         line.erase(line.find_last_not_of("\r\n ") + 1);  // strip newline
         
         if (!line.empty()) {
            classes.push_back(line);
         }
      }
   }

   classFile.close();

   return classes;
}

/*
downmix audio tensor
input tensor must be shape (batch, channels, time)
*/
torch::Tensor DeepModel::downmix(const torch::Tensor audioBatch) {
   assert (audioBatch.dim() == 3);
   // if channel dim is already 1, return
   if (audioBatch.sizes()[1] == 1 ){
      return audioBatch;
   }

   // take the mean over the channel dimension
   torch::Tensor downmixedAudio =  audioBatch.mean(1, true);
   return downmixedAudio;
}

/**
 @brief: pads an audio tensor with shape (samples,)
    with the necessary zeros and then reshapes to (batch, 1, chunkLen)
@params:
   torch::Tensor audio: MONO audio tensor with shape (samples,)
@returns:
   torch::Tensor reshapedAudio: audio tensor with shape (batch, 1, chunkLen)
*/
torch::Tensor DeepModel::padAndReshape(torch::Tensor audio){
   audio = audio[0][0];
   auto length = audio.sizes()[0];
   
   // RIGHT: pad with zeros to meet length
   int newLength = ceil((double)length/chunkLen) * chunkLen;
   int padLength = newLength - length;
   
   torch::Tensor reshapedAudio;
   if (padLength == 0){
      reshapedAudio = torch::Tensor(audio);
   } else {
      torch::Tensor padTensor = torch::zeros({padLength});
      reshapedAudio = torch::cat({audio, padTensor});
   }

   reshapedAudio = reshapedAudio.reshape({-1, 1, chunkLen});
   return reshapedAudio;
}

/* 
TODO: the already-compiled models return log-probabilities. 
this should be fixed on the python side though, to keep this class architecture agnostic. 
@brief: forward pass through the model and get raw class probabilities as output
@params:
   torch::Tensor audioBatch: batch of mono audio with shape (batch, 1, chunkLen)
@returns:
   torch::Tensor probits: per-class probabilities with shape (batch, n_classes)
*/
torch::Tensor DeepModel::modelForward(const torch::Tensor inputAudio){
   std::vector<torch::jit::IValue> inputs;
   inputs.push_back(inputAudio);

   auto sz = inputAudio.sizes().vec();
   // for (int i = 0; i < sz[3]; i++){
   //    std::cout<<inputAudio[0][0][0][i]<<std::endl;
   // }
   
   // get class probabilities
   auto output = jitModel.forward({inputs}).toTensor();
   return output;
}

