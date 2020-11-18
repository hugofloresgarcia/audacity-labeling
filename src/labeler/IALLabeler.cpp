
//
//  IALLabeler.cpp
//  Audacity
//
//  Created by Jack Wiig on 4/28/20.
//  Modified by Hugo Flores on 7/31/20.

#include <iostream>
#include <cmath>

#include "IALLabeler.hpp"

#include "IALAudioFrame.hpp"
#include "WaveTrack.h"
#include "../WaveClip.h"
#include "../Track.h"
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

void IALLabeler::labelTracks()
{
}

void IALLabeler::updateTracks()
{
    TrackList &trackList = TrackList::Get(const_cast<AudacityProject&>(project));
    for (Track *track : trackList)
    {
        if (dynamic_cast<WaveTrack *>(track) != nullptr)
        {
            std::shared_ptr<WaveTrack> waveTrack = std::dynamic_pointer_cast<WaveTrack>(track->SharedPointer());

            Track *leader = *trackList.FindLeader(track);
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
        }
    }
}
