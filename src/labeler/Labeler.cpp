
//
//  Labeler.cpp
//  Audacity
//
//  Created by Jack Wiig on 4/28/20.
//  Modified by Hugo Flores on 7/31/20.

#include <iostream>
#include <cmath>

#include <essentia/essentia.h>
#include <essentia/algorithmfactory.h>
#include <torch/script.h>
#include <wx/textfile.h>

#include "Labeler.hpp"
#include "WaveTrack.h"
#include "../FileNames.h"
#include "../commands/CommandContext.h"
#include "../LabelTrack.h"
#include "../ProjectHistory.h"

#include "Model.hpp"


using namespace essentia::standard;

struct AudacityLabel {
    int start;
    int end;
    std::string label;
    
    AudacityLabel(float start, float end, std::string label) : start(start), end(end), label(label) {};
    
    std::string toStdString() {
        return std::to_string(start) + "\t" + std::to_string(end) + "\t" + label;
    }
};


std::vector<std::vector<essentia::Real>> loadAudioThroughVGGish(const std::string &filepath) {

    essentia::init();
    essentia::Real sampleRate = 16000;
    AlgorithmFactory &factory = AlgorithmFactory::instance();

    std::string vggishModelPath = wxFileName(FileNames::ResourcesDir(), wxT("vggish.pb")).GetFullPath().ToStdString();

    Algorithm *loader = factory.create("MonoLoader",
                                       "filename", filepath,
                                       "sampleRate", sampleRate);
                                           
    Algorithm *vggish = factory.create("TensorflowPredictVGGish",
                                       "input", "vggish/input_features",
                                       "output", "vggish/fc2/BiasAdd",
                                       "patchHopSize", 0,
                                       "graphFilename", vggishModelPath);
    
    // Connect Algorithms
    // EasyLoader -> VGGish
    std::vector<essentia::Real> audioBuffer;
    loader->output("audio").set(audioBuffer);
    vggish->input("signal").set(audioBuffer);
    
    // VGGish -> Output
    std::vector<std::vector<essentia::Real>> embeddings;
    vggish->output("predictions").set(embeddings);
    
    
    // Run Algos
    loader->compute();
    vggish->compute();
    
    delete loader;
    delete vggish;
    factory.shutdown();
    essentia::shutdown();

    return embeddings;   
}


std::vector<std::string>loadInstrumentList(const std::string &filepath) {
    std::vector<std::string> instruments;
    std::ifstream instrumentFile(filepath);

    if (instrumentFile.is_open()) {
        while (instrumentFile.good()) {
            std::string line;
            getline(instrumentFile, line);
            line.erase(line.find_last_not_of("\r\n ") + 1);  // strip newline
            
            if (!line.empty()) {
                instruments.push_back(line);
            }
        }
    }

    instrumentFile.close();
    
    return instruments;
}


torch::jit::script::Module loadModel(const std::string &filepath) {
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


std::vector<std::string> labelTrackEmbeddings(const std::vector<std::vector<essentia::Real>> &embeddings, const std::vector<std::string> &instruments) {
    
    // Load Model
    std::string modelPath = wxFileName(FileNames::ResourcesDir(), wxT("classifier.pt")).GetFullPath().ToStdString();
    torch::jit::script::Module classifierModel = loadModel(modelPath);
    
    int embeddingLength = (int) embeddings[0].size();
    int numAdjacent = 4;
    std::vector<std::string> predictions;
    
    // If there are fewer than 2*numAdjacent embeddings, then there is not enough to predict on.
    if (embeddings.size() < 2*numAdjacent) {
        return predictions;
    }
    
    for (int i = 0; i < embeddings.size(); i++) {
        std::vector<at::Tensor> slice;

        // Collect a batch of the [i-4, i+4) examples into a vector, and use that as input to the model
        for (int j = i - numAdjacent; j < i + numAdjacent; j++) {

            // At the start, use examples from the end
            if (j < 0) {
                int endIdx = (int) embeddings.size() - abs(j);
                slice.emplace_back(at::tensor(embeddings[endIdx]));
            }

            // If at end, use examples from the start
            else if (j >= embeddings.size()) {
                int startIdx = j - (int) embeddings.size();
                slice.emplace_back(at::tensor(embeddings[startIdx]));
            }

            // Otherwise, use the example
            else {
                slice.emplace_back(at::tensor(embeddings[j]));
            }
        }

        at::Tensor inputSlice = torch::reshape(torch::stack(slice), {1, embeddingLength, 2*numAdjacent});
        int encodedInstrument = torch::argmax(classifierModel.forward({inputSlice}).toTensor()).item<int>();
        
        predictions.emplace_back(instruments[encodedInstrument]);

    }

    return predictions;
}


std::vector<AudacityLabel> coalesceLabels(const std::vector<AudacityLabel> &labels) {
    std::vector<AudacityLabel> coalescedLabels;
    
    if (labels.size() == 0) {
        return coalescedLabels;
    }
    
    int startIdx = 0;
    for (int i = 0; i < labels.size(); i++) {
        
        // Iterate over a range until a label differs from the previous label AND
        // previous label end time is same as the current label's start time
        if (labels[i].label != labels[startIdx].label
            && labels[i - 1].end != labels[i].start) {
            coalescedLabels.push_back(AudacityLabel(labels[startIdx].start,
                                                    labels[i - 1].end,
                                                    labels[startIdx].label));
            startIdx = i;
        }
    }
    
    coalescedLabels.push_back(AudacityLabel(labels[startIdx].start,
                                            labels[labels.size() - 1].end,
                                            labels[startIdx].label));
    
    return coalescedLabels;
}


std::map<std::string, std::vector<AudacityLabel>> createAudacityLabels(const std::vector<std::string> &embeddingLabels) {
    
    std::map<std::string, std::vector<AudacityLabel>> audacityLabels;
    
    float frameLength = 0.96;
    float windowStartInSeconds = 0;
    for (const std::string &label : embeddingLabels) {
        
        // Round to the nearest second
        int roundedWindowStart = round(windowStartInSeconds);
        int roundedWindowEnd = round(windowStartInSeconds + frameLength);
        
        // If they are equal, don't add the Audacity Label
        if (roundedWindowStart == roundedWindowEnd) {
            windowStartInSeconds += frameLength;
            continue;
        }
        
        AudacityLabel windowedLabel = AudacityLabel(roundedWindowStart, roundedWindowEnd, label);

        // Add instrument label to map if it's not in it
        if (audacityLabels.find(label) == audacityLabels.end()) {
            audacityLabels.insert(std::pair<std::string, std::vector<AudacityLabel>>(label, std::vector<AudacityLabel>()));
        }
        
        audacityLabels[label].push_back(windowedLabel);
        
        windowStartInSeconds += frameLength;
    }
    
    // Coalesce pushed labels
    for (auto &labelSet : audacityLabels) {
        labelSet.second = coalesceLabels(labelSet.second);
    }
    
    return audacityLabels;
}

void IALLabeler::LabelTrack(const CommandContext &context, const std::string &filepath) {

    auto &project = context.project;
    auto &trackFactory = TrackFactory::Get( project );
    auto &tracks = TrackList::Get( project );
    
    std::string classifierLabelsPath = wxFileName(FileNames::ResourcesDir(), wxT("classifier_instruments.txt")).GetFullPath().ToStdString();
    
    // Load audio file as VGGish embeddings
    std::vector<std::vector<essentia::Real>> embeddings = loadAudioThroughVGGish(filepath);
    
    // Load the classification classes into a vector
    std::vector<std::string> instruments = loadInstrumentList(classifierLabelsPath);
    
    // Using the VGGish embeddings and output classes, label the embeddings
    std::vector<std::string> trackLabels = labelTrackEmbeddings(embeddings, instruments);
    
    // In the event the song was too short to label, don't label it.
    if (trackLabels.empty()) {
        return;
    }
    
    // Given embedding-wise labels, create a collection of trackwise labels for each instrument
    std::map<std::string, std::vector<AudacityLabel>> labels = createAudacityLabels(trackLabels);
    
    // Add a Label Track for each class of labels
    for (auto &labelTrack : labels) {
        wxString labelFileName = wxFileName(FileNames::DataDir(), labelTrack.first + ".txt").GetFullPath();
        wxTextFile labelFile(labelFileName);
        
        // In the event of a crash, the file might still be there. If so, clear it out and get it ready for reuse. Otherwise, create a new one.
        if (labelFile.Exists()) {
            labelFile.Clear();
        }
        else {
            labelFile.Create();
        }
        labelFile.Open();
        
        // Write each timestamp to file
        for (auto &label : labelTrack.second) {
            labelFile.AddLine(wxString(label.toStdString()));
        }
        
        // Create a new LabelTrack and add it to the project
        auto newTrack = trackFactory.NewLabelTrack();
        newTrack->SetName(wxString(labelTrack.first));
        newTrack->Import(labelFile);
        tracks.Add(newTrack);
        
        // Record Adding the Track
        ProjectHistory::Get(project).PushState(XO("Automatically Imported '%s' Labels for '%s'").Format(labelTrack.first, filepath), XO("Auto-Imported Labels"));
        
        labelFile.Close();
        wxRemove(labelFileName);
    }
}
