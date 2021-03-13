#include "ClassificationModel.h"


/*
@briefs: forward pass through the model and get a list of classes with the highest probabilities for every instance in the batch. 
@params: 
   torch::Tensor audioBatch: batch of mono audio with shape (batch, 1, chunkLen)
   float confidenceThreshold: probabilities under this value will be labeled 'not-sure'
@returns:  
   std::vector<std::string> predictions: list of class predictions for every instance in the batch
*/
std::vector<std::string> ClassificationModel::predictFromAudioFrame(const torch::Tensor audioBatch, float confidenceThreshold = 0.3)
{
   auto probits = predict(audioBatch);
   auto [confidences, indices] = probits.max(1, false);

   std::vector<std::string> predictions = constructLabelsFromProbits(confidences, indices, confidenceThreshold);
   return predictions;
}

std::vector<std::string> ClassificationModel::predictFromAudioSequence(const torch::Tensor audioSequence, float confidenceThreshold = 0.3)
{
   // probits should be a tensor size (seq, batch, probit)
   auto probits = predict(audioSequence);

   // ASSUME batch size is 1 
   assert(probits.sizes()[1] == 1);
   auto [confidences, indices] = probits.max(-1, false);

   confidences = confidences.squeeze(-1);
   indices = indices.squeeze(-1);

   std::vector<std::string>predictions = constructLabelsFromProbits(confidences, indices, confidenceThreshold);

   return predictions;
}

std::vector<std::string> ClassificationModel::constructLabelsFromProbits(const torch::Tensor confidences, 
                                                       const torch::Tensor indices, 
                                                       float confidenceThreshold)
{
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
         prediction = classes.at(int(idx));
      }
      
      predictions.push_back(prediction);
   }
   return predictions;
}

// this is not actually a proper test and will be deleted
void ClassificationModel::modelTest(torch::Tensor inputAudio){
   try{
      std::cout << " downmixing audio" << "\n";
      // inputAudio = downmix(inputAudio);

      std::cout << "doing predictions:" << "\n";
      std::vector<std::string> predictions = predictFromAudioFrame(inputAudio);
      
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





torch::Tensor ClassificationModel::predict(const torch::Tensor inputAudio, bool addSoftmax)
{
   auto output = modelForward(inputAudio);
   // adding a softmax here
   auto probits = torch::softmax(output, -1);
   std::cout << probits << '\n';
   return probits;
}
