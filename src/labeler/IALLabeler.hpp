//
//  IALLabeler.hpp
//  Audacity
//
//  Created by Jack Wiig on 4/28/20.
//

#ifndef IALLabeler_hpp
#define IALLabeler_hpp

#include <stdio.h>
#include <unordered_map>

#include "IALAudioFrame.hpp"

class CommandContext;
class SampleBuffer;
class WaveTrack;


class IALLabeler {
    const CommandContext &context;

public:
    IALLabeler(const CommandContext &context);
        
    void labelTracks();
    
private:
    std::vector<SampleBuffer> fetchProjectAudio();
    std::unordered_map<int, IALAudioFrameTrack> tracks;
};


#endif
