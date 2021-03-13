//
//  IALLabeler.hpp
//  Audacity
//
//  Created by Jack Wiig on 4/28/20.
//

#ifndef IALLabeler_hpp
#define IALLabeler_hpp

#include <stdio.h>
#include <memory>
#include <map>

#include "ClientData.h"
#include "../Track.h"
#include "IALAudioFrame.hpp"
#include "ClassificationModel.h"

class LabelTrack;
class ClassifierModel;

class IALLabeler
    : public ClientData::Base,
      public std::enable_shared_from_this<IALLabeler>
{
    
public:
    ClassificationModel classifier;
    
    // Get static instance
    static IALLabeler &Get(AudacityProject &project);
    static const IALLabeler &Get(const AudacityProject &project);
    
    // Constructor
    IALLabeler(AudacityProject &project);
    
    // Disable the copy constructors
    IALLabeler(const IALLabeler &that) = delete;
    IALLabeler &operator= (const IALLabeler&) = delete;
    
    // Disable the move constructors
    IALLabeler(IALLabeler &&that) = delete;
    IALLabeler& operator= (IALLabeler&&) = delete;
    
    void labelTrack(Track* track);
    void labelTracks();
    
private:
    AudacityProject &project;
    // assumes tracks have already been labeled
    void arrangeTracks();
    
    std::map<TrackId, IALAudioFrameCollection> tracks;
};

#endif
