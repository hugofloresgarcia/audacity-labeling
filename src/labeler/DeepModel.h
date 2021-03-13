#ifndef DeepModel_h
#define DeepModel_h

#include <iostream>
#include <cmath>
#include <assert.h>
#include <torch/script.h>


class DeepModel {
   torch::jit::script::Module loadModel(const std::string &filepath);
   
   const int chunkLen = 48000;

   public:
      
      std::vector<std::string> loadClasslist(const std::string &filepath);
      DeepModel(){};
      DeepModel(const std::string &modelPath, const std::string &classlistPath);

      const std::vector<std::string> &getClasslist() {return classes;}
      const int getChunkLen() {return chunkLen;}

      torch::Tensor downmix(const torch::Tensor audioBatch);
      torch::Tensor padAndReshape(const torch::Tensor audio);

      torch::Tensor modelForward(const torch::Tensor inputAudio);

   protected:
      torch::jit::script::Module jitModel;
      std::vector<std::string> classes;
};

#endif
