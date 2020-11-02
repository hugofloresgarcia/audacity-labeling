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
#include <unordered_map>

#include "ClientData.h"
#include "IALAudioFrame.hpp"

class LabelTrack;
class IALTrackAnalysis;

class IALLabeler
    : public ClientData::Base,
      public std::enable_shared_from_this<IALLabeler>
{
    
public:
    // Get static instance
    static IALLabeler &Get(AudacityProject &project);
    static const IALLabeler &Get(const AudacityProject &project);
    
    // Constructor
    IALLabeler(const AudacityProject &project);
    
    // Disable the copy constructors
    IALLabeler(const IALLabeler &that) = delete;
    IALLabeler &operator= (const IALLabeler&) = delete;
    
    // Disable the move constructors
    IALLabeler(IALLabeler &&that) = delete;
    IALLabeler& operator= (IALLabeler&&) = delete;
    
    void labelTracks();
    
private:
    const AudacityProject &project;
    
    std::unordered_map<int, IALAudioFrameCollection> tracks;
//    std::vector<IALTrackAnalysis> tracks;
};


class IALTrackAnalysis
{
public:
    std::string label();
    IALAudioFrameCollection frames;
    
    std::weak_ptr<const LabelTrack> labelTrack;
};

#endif
