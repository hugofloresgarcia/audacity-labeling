//
//  Labeler.hpp
//  Audacity
//
//  Created by Jack Wiig on 4/28/20.
//

#ifndef Labeler_hpp
#define Labeler_hpp

#include <stdio.h>

class CommandContext;
class SampleBuffer;

class IALLabeler {
    const CommandContext &context;

public:
    IALLabeler(const CommandContext &context);
        
    void labelTracks();
    
private:
    std::vector<SampleBuffer> fetchProjectAudio();
};

namespace IALLabelerSpace {

void LabelTrack(const CommandContext &context, const std::string &audioFilePath);
// std::ofstream LabelLogger;
void LabelTracks(const CommandContext &context);
}


#endif
