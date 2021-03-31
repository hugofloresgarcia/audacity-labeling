
//
//  IALLabeler.cpp
//  Audacity
//
//  Created by Jack Wiig on 4/28/20.
//  Modified by Hugo Flores on 7/31/20.

#include <iostream>
#include <cmath>

#include "IALLabeler.hpp"
#include "ProjectHistory.h"

#include <wx/textfile.h>

#include "IALAudioFrame.hpp"
#include "WaveTrack.h"
#include "../WaveClip.h"
#include "../Track.h"
#include "../TrackUtilities.h"
#include "../LabelTrack.h"
#include "../ViewInfo.h"
#include "../SampleFormat.h"


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

IALLabeler::IALLabeler(AudacityProject &project)
    : classifier(kModelPath, kInstrumentListPath), project(project), tracks(std::map<TrackId, IALAudioFrameCollection>())
{
}

// # pragma mark Private

void IALLabeler::labelTracks()
{   
    // try{
    TrackList &tracklist = TrackList::Get(const_cast<AudacityProject&>(project));
    const auto& playableTracks = tracklist.Any<PlayableTrack>();

    for (Track *track : playableTracks ){ 
        // we want to arrange the tracks until after we're done iterating through them
        labelTrack(track, false);
    }
    // arrange the tracks
    arrangeTracks();
}

// because we're only allowed to move the tracks either once down or up, 
/// or all the way to the bottom or top, we'll start from the 
// topmost track (iterating through  all playable tracks) 
void IALLabeler::arrangeTracks(){
    // skip for now
    return; 

    TrackList &tracklist = TrackList::Get(const_cast<AudacityProject&>(project));
    const auto& playableTracks = tracklist.Any<PlayableTrack>();

    if (tracklist.size() > 2){
        for (auto framePair: tracks){
            TrackId leaderId = framePair.first;
            IALAudioFrameCollection& frameCollection = framePair.second;

            Track* leader = tracklist.FindById(leaderId);
            TrackId labelTrackId = frameCollection.labelTrack->GetId();
            Track* labelTrack = tracklist.FindById(labelTrackId);

            // start with the leader, move all the way to the bottom
            TrackUtilities::DoMoveTrack(project, leader, TrackUtilities::MoveChoice::OnMoveBottomID);

            // then move the label track all the to the bottom
            TrackUtilities::DoMoveTrack(project, labelTrack, TrackUtilities::MoveChoice::OnMoveBottomID);

            // auto history = ;
            ProjectHistory::Get( project ).PushState(XO("Moved Labeled Track Pair"), XO("Move Labeled"));

        }
    } 
}

void IALLabeler::labelTrack(Track* track, bool arrange)
{   
    TrackList &tracklist = TrackList::Get(const_cast<AudacityProject&>(project));
    const auto& playableTracks = tracklist.Any<PlayableTrack>();

    if (dynamic_cast<WaveTrack *>(track) != nullptr)
    {
        std::shared_ptr<WaveTrack> waveTrack = std::dynamic_pointer_cast<WaveTrack>(track->SharedPointer());

        Track *leader = *tracklist.FindLeader(track);
        TrackId leaderID = leader->GetId();
        std::shared_ptr<WaveTrack> leaderTrack = std::dynamic_pointer_cast<WaveTrack>(leader->SharedPointer());
        
        // find out if we have labeled this track before
        // if we haven't, create a new entry 
        auto pair = tracks.find(leaderID);
        if (pair == tracks.end())
        {
            tracks.insert(std::make_pair(leaderID, IALAudioFrameCollection(classifier, leaderTrack)));
            pair = tracks.find(leaderID);
        }

        // grab the frame collection, and add a new channel to it if necessary
        IALAudioFrameCollection& frameCollection = tracks.find(leaderID)->second;
        frameCollection.addChannel(std::weak_ptr<WaveTrack>(waveTrack));

        // update collection length
        frameCollection.updateCollectionLength();

        // update the labels
        frameCollection.labelAllFrames(project);

        // auto history = ;
        ProjectHistory::Get( project ).PushState(XO("Labeled Track"), XO("Label"));
    }

    if (arrange){
        arrangeTracks();
    }
}

void IALLabeler::separateTrack(Track* track)
{
    TrackList &tracklist = TrackList::Get(const_cast<AudacityProject&>(project));
    const auto& playableTracks = tracklist.Any<PlayableTrack>();

    static const std::string kSeparationModelPath = wxFileName(FileNames::ResourcesDir(), wxT("separation-model.pt")).GetFullPath().ToStdString();
    static const std::string kSeparationInstrumentListPath = wxFileName(FileNames::ResourcesDir(), wxT("separation-instruments.txt")).GetFullPath().ToStdString();

    int sampleRate = 8000;
    DeepModel separationModel(kSeparationModelPath, kSeparationInstrumentListPath);
    separationModel.setChunkLen(8000);

    if (dynamic_cast<WaveTrack *>(track) != nullptr)
    {
        std::shared_ptr<WaveTrack> waveTrack = std::dynamic_pointer_cast<WaveTrack>(track->SharedPointer());

        Track *leader = *tracklist.FindLeader(track);
        TrackId leaderID = leader->GetId();
        std::shared_ptr<WaveTrack> leaderTrack = std::dynamic_pointer_cast<WaveTrack>(leader->SharedPointer());
        
        sampleFormat originalFormat = leaderTrack->GetSampleFormat();
        double originalSampleRate = leaderTrack->GetRate();
        size_t actualLength = leaderTrack->TimeToLongSamples(leaderTrack->GetEndTime()).as_size_t();
        
        // SampleBuffer channelBuffer = SampleBuffer(desiredLength, format);
        // AudacityProject *project = leaderTrack->GetOwner()->GetOwner();
        SampleBlockFactoryPtr sbFactory = WaveTrackFactory::Get(project).GetSampleBlockFactory();

        // copy channel's samples into buffer
        SampleBuffer buffer(actualLength, originalFormat);
        // fill up the buffer
        leaderTrack->Get(buffer.ptr(), originalFormat, 0, actualLength);

        // make a separate clip where we will do the necessary conversions 
        WaveClip conversionClip(sbFactory, originalFormat, leaderTrack->GetRate(), leaderTrack->GetWaveColorIndex());

        // fill the clip with our buffer
        conversionClip.Append(buffer.ptr(), originalFormat, actualLength);
        conversionClip.Flush();
        // conversionClip.SetSamples(buffer.ptr(), originalFormat, start, actualLength);

        // do the conversions
        conversionClip.ConvertToSampleFormat(floatSample);
        conversionClip.Resample(sampleRate);

        // copy channel's samples into buffer
        SampleBuffer outBuffer(conversionClip.GetNumSamples().as_size_t(), floatSample);
        conversionClip.GetSamples(outBuffer.ptr(), floatSample, conversionClip.GetStartSample(), conversionClip.GetNumSamples().as_size_t());
        
        torch::Tensor bufTensor = torch::from_blob(outBuffer.ptr(),
                                                       conversionClip.GetNumSamples().as_size_t(),
                                                       torch::TensorOptions().dtype(torch::kFloat32));

        torch::Tensor channelTensor = bufTensor.clone();

        buffer.Free();
        
        // expand channel and batch dimes
        channelTensor = separationModel.padAndReshape(channelTensor.view({1, 1, -1}));
        auto sz = channelTensor.sizes();

        torch::Tensor output = separationModel.modelForward(channelTensor);
        
        // this could be a risky operation?
        // view as two channels of separate sources
        output = output.view({1, 2, -1});
        torch::Tensor source = output[0][0];
        torch::Tensor source2 = output[0][1];

        for (auto source: std::vector<torch::Tensor>{source, source2})
        {
            source = source.clone() / 10000;
            auto sourceLength = source.sizes()[0];
            SampleBuffer sourceBuffer(source.sizes()[0], floatSample);

            // sourceBuffer()
            auto sourcePtr = source.accessor<float, 1>();

            // ooo copying by hand
            // CopySamples((samplePtr)source.to(torch::kFloat32).data_ptr<float>(), 
            //               floatSample, sourceBuffer.ptr(), floatSample, sourceLength)
            // float* s = source.contiguous().to(torch::kFloat32).data_ptr<float>();
            // float* d = (float*)sourceBuffer.ptr();
            // for (int i = 0; i < sourceLength; i++, d += 1, s += 1)
            //     *d = *s;

            // CopySamples(source.data_ptr<float>(), floatSample, sourceBuffer.ptr(), floatSample, sourceLength);
            
            // make a separate clip where we will do the necessary conversions 
            auto newTrack = std::make_shared<WaveTrack>(sbFactory, floatSample, 8000);

            // fill the clip with our buffer
            // newTrack->Append(sourceBuffer.ptr(), floatSample, sourceLength);
            newTrack->Append((samplePtr)source.to(torch::kFloat32).data_ptr<float>(),
                              floatSample, sourceLength);
            newTrack->Flush();

            tracklist.Add(newTrack);
            // auto history = ;
        }

        ProjectHistory::Get( project ).PushState(XO("Separated Track"), XO("SourceSep"));
    }

}