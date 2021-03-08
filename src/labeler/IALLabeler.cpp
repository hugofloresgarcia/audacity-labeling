
//
//  IALLabeler.cpp
//  Audacity
//
//  Created by Jack Wiig on 4/28/20.
//  Modified by Hugo Flores on 7/31/20.

#include <iostream>
#include <cmath>

#include "IALLabeler.hpp"

#include <wx/textfile.h>

#include "IALAudioFrame.hpp"
#include "WaveTrack.h"
#include "../WaveClip.h"
#include "../Track.h"
#include "../LabelTrack.h"
#include "../FileNames.h"
#include "../ViewInfo.h"


#pragma mark ClientData Initialization

static const std::string kModelPath = wxFileName(FileNames::ResourcesDir(), wxT("ial-model.pt")).GetFullPath().ToStdString();
static const std::string kInstrumentListPath = wxFileName(FileNames::ResourcesDir(), wxT("ial-instruments.txt")).GetFullPath().ToStdString();

/**
 @brief An anonymous function that initializes an instance of IALLabeler.
 @warning Do not call this function directly.
 @discussion This function is used alongside the RegisteredFactory class. It is passed to the default constructor, where it is called
 and the result (a smart pointer to IALLabeler) is stored in the list of AttachedObjects. From here, we can fetch it from anything using
 the project.
 */
static auto IALLabelerFactory = [](AudacityProject &project)
{
    return std::make_shared<IALLabeler>(project);
};

/**
 @brief This is a static RegisteredFactory instance initialized with the anonymous function IALLabelerFactory
 @discussion Essentially, what's happening here is that this labelerKey will call the constructor to RegisteredFactory and register
 this instance of IALLabeler under the key labelerKey. Any fetch calls to AttachedObjects::Get with labelerKey will return this instance.
 The constructor for RegisteredFactory appends it to the list of factories.
 */
static const AudacityProject::AttachedObjects::RegisteredFactory labelerKey
{
    IALLabelerFactory
};

#pragma mark Instance Getter

IALLabeler &IALLabeler::Get(AudacityProject &project)
{
    return project.AttachedObjects::Get<IALLabeler>(labelerKey);
}

const IALLabeler &IALLabeler::Get(const AudacityProject &project)
{
    return Get(const_cast<AudacityProject &>(project));
}

#pragma mark Initializer

IALLabeler::IALLabeler(const AudacityProject &project)
    : classifier(kModelPath, kInstrumentListPath), project(project), tracks(std::map<TrackId, IALAudioFrameCollection>())
{
    updateTracks();
}

# pragma mark Private

struct AudacityLabel {
    int start;
    int end;
    std::string label;
    
    AudacityLabel(float start, float end, std::string label) : start(start), end(end), label(label) {};
    
    std::string toStdString() {
        return std::to_string(start) + "\t" + std::to_string(end) + "\t" + label;
    }
};

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
std::vector<AudacityLabel> createAudacityLabels(const std::vector<std::string> &embeddingLabels) {
    
    std::vector<AudacityLabel> audacityLabels;
    
    float frameLength = 1.0;
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

        audacityLabels.push_back(windowedLabel);
        
        windowStartInSeconds += frameLength;
    }

    audacityLabels = coalesceLabels(audacityLabels);
    
    return audacityLabels;
}

bool trackInTrackList(TrackList& tracklist, std::shared_ptr<LabelTrack> track){
    // Track *leader = *tracklist.FindLeader(track);
    TrackId id = track->GetId();

    for (Track *other: tracklist){
        TrackId otherId = other->GetId();

        if (otherId == id){
            return true;
        }
    }
    return false;
}

//todo: improve me (O(n^2) for number of tracks)
void IALLabeler::labelTracks()
{   
    TrackList &tracklist = TrackList::Get(const_cast<AudacityProject&>(project));
    const auto& playableTracks = tracklist.Any<PlayableTrack>();

    for (Track *track : playableTracks ){ 
        if (dynamic_cast<WaveTrack *>(track) != nullptr)
        {
            std::shared_ptr<WaveTrack> waveTrack = std::dynamic_pointer_cast<WaveTrack>(track->SharedPointer());

            Track *leader = *tracklist.FindLeader(track);
            TrackId leaderID = leader->GetId();
            std::shared_ptr<WaveTrack> leaderTrack = std::dynamic_pointer_cast<WaveTrack>(leader->SharedPointer());
            
            auto pair = tracks.find(leaderID);

            if (pair == tracks.end())
            {
                tracks.insert(std::make_pair(leaderID, IALAudioFrameCollection(classifier, leaderTrack)));
                pair = tracks.find(leaderID);
            }

            IALAudioFrameCollection& frameCollection = tracks.find(leaderID)->second;
            frameCollection.addChannel(std::weak_ptr<WaveTrack>(waveTrack));

            std::vector<std::string> predictions;
            for (auto frame : frameCollection.audioFrames){
                std::string prediction = frame.label();
                predictions.emplace_back(prediction);
            }

            // TODO: find a better way to create label tracks than this
            if (!predictions.empty()){
                std::vector<AudacityLabel> labels = createAudacityLabels(predictions);

                wxString labelFileName = wxFileName(FileNames::DataDir(), predictions[0] + ".txt").GetFullPath();
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
                for (auto &label : labels) {
                    labelFile.AddLine(wxString(label.toStdString()));
                }

                // std::shared_ptr<LabelTrack> newTrack = frameCollection.labelTrack;
                
                auto labelpair = id2labels.find(leaderID);
                std::shared_ptr<LabelTrack> newTrack;
                if (labelpair == id2labels.end()){ // if we haven't registered this yet
                    // Create a new LabelTrack and add it to the project
                    newTrack = std::make_shared<LabelTrack>();
                    // store the track pointer for later use
                    id2labels.insert(std::make_pair(leaderID, newTrack));
                } else {
                    newTrack = id2labels[leaderID];
                }
                newTrack->SetName(wxString(predictions[0]));
                newTrack->Import(labelFile);

                if (!trackInTrackList(tracklist, newTrack)) {
                    tracklist.Add(newTrack);
                }
                
                labelFile.Close();
                wxRemove(labelFileName);
            }
        }
    }
}


void IALLabeler::updateTracks()
{
    // TrackList &trackList = TrackList::Get(const_cast<AudacityProject&>(project));
    // for (Track *track : trackList)
    // {
    //     if (dynamic_cast<WaveTrack *>(track) != nullptr)
    //     {
    //         std::shared_ptr<WaveTrack> waveTrack = std::dynamic_pointer_cast<WaveTrack>(track->SharedPointer());

    //         Track *leader = *trackList.FindLeader(track);
    //         TrackId leaderID = leader->GetId();
    //         std::shared_ptr<WaveTrack> leaderTrack = std::dynamic_pointer_cast<WaveTrack>(leader->SharedPointer());
            
    //         auto pair = tracks.find(leaderID);

    //         if (pair == tracks.end())
    //         {
    //             tracks.insert(std::make_pair(leaderID, IALAudioFrameCollection(classifier, leaderTrack)));
    //             pair = tracks.find(leaderID);
    //         }

    //         IALAudioFrameCollection& frameCollection = tracks.find(leaderID)->second;
    //         frameCollection.addChannel(std::weak_ptr<WaveTrack>(waveTrack));
    //     }
    // }
}




// TrackList &trackList = TrackList::Get(const_cast<AudacityProject&>(project));
//     for (Track *track : trackList)
//     {
//         if (dynamic_cast<WaveTrack *>(track) != nullptr)
//         {
//             std::shared_ptr<WaveTrack> waveTrack = std::dynamic_pointer_cast<WaveTrack>(track->SharedPointer());

//             Track *leader = *trackList.FindLeader(track);
//             TrackId leaderID = leader->GetId();
//             std::shared_ptr<WaveTrack> leaderTrack = std::dynamic_pointer_cast<WaveTrack>(leader->SharedPointer());
            
//             auto pair = tracks.find(leaderID);

//             if (pair == tracks.end())
//             {
//                 tracks.insert(std::make_pair(leaderID, IALAudioFrameCollection(classifier, leaderTrack)));
//                 pair = tracks.find(leaderID);
//             }

//             IALAudioFrameCollection& frameCollection = tracks.find(leaderID)->second;
//             frameCollection.addChannel(std::weak_ptr<WaveTrack>(waveTrack));

//             std::vector<std::string> predictions;
//             for (auto frame : frameCollection.audioFrames){
//                 std::string prediction = frame.label();
//                 predictions.emplace_back(prediction);
//             }

//             if (!predictions.empty()){
//                 std::vector<AudacityLabel> labels = createAudacityLabels(predictions);

//                 wxString labelFileName = wxFileName(FileNames::DataDir(), labels[0] + ".txt").GetFullPath();
//                 wxTextFile labelFile(labelFileName);
                
//                 // In the event of a crash, the file might still be there. If so, clear it out and get it ready for reuse. Otherwise, create a new one.
//                 if (labelFile.Exists()) {
//                     labelFile.Clear();
//                 }
//                 else {
//                     labelFile.Create();
//                 }
//                 labelFile.Open();
                
//                 // Write each timestamp to file
//                 for (auto &label : labels) {
//                     labelFile.AddLine(wxString(label.toStdString()));
//                 }
                
//                 // Create a new LabelTrack and add it to the project
//                 std::shared_ptr<LabelTrack> newTrack = std::make_shared<LabelTrack>();
//                 newTrack->SetName(wxString(labels[0]));
//                 newTrack->Import(labelFile);
//                 trackList.Add(newTrack);
                
//                 labelFile.Close();
//                 wxRemove(labelFileName);
//                 }
//             }
//         } 
//     }