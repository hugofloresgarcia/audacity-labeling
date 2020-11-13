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
#include <wx/event.h>

#include "ClientData.h"
#include "../Track.h"
#include "IALAudioFrame.hpp"

class LabelTrack;
class IALTrackAnalysis;

class IALLabeler
    : public ClientData::Base,
      public wxEvtHandler,
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
    
    void processSelectedRegion(wxCommandEvent &event);
    
    std::map<TrackId, IALAudioFrameCollection> tracks;
//    std::vector<IALTrackAnalysis> tracks;
};

#endif
