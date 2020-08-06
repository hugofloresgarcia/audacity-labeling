//
//  Labeler.hpp
//  Audacity
//
//  Created by Jack Wiig on 4/28/20.
//

#ifndef Labeler_hpp
#define Labeler_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>

class CommandContext;

namespace IALLabeler {
    void LabelTrack(const CommandContext &context, const std::string &audioFilePath);
}


#endif /* PythonBridge_hpp */
