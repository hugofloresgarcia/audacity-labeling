
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
#include "../commands/CommandContext.h"
#include "../WaveClip.h"
#include "../Track.h"


#pragma mark IALLabeler Class Definition

IALLabeler::IALLabeler(const CommandContext &context)
    : context(context), tracks(std::unordered_map<int, IALAudioFrameTrack>()) {}

void IALLabeler::labelTracks()
{
    TrackList &tracklist = TrackList::Get(context.project);
    
    for (Track *track : tracklist)
    {
        if (dynamic_cast<WaveTrack *>(track) != nullptr)
        {
            std::shared_ptr<WaveTrack> waveTrack = std::dynamic_pointer_cast<WaveTrack>(track->SharedPointer());
            
            int trackIdx = waveTrack->GetIndex();
            auto pair = tracks.find(trackIdx);
            
            if (pair == tracks.end())
            {
                tracks.insert(std::make_pair(trackIdx, IALAudioFrameTrack(std::weak_ptr<WaveTrack>(waveTrack))));
                
                pair = tracks.find(trackIdx);
            }
            
            IALAudioFrameTrack& frameTrack = tracks.find(trackIdx)->second;
            
            for (std::shared_ptr<IALAudioFrame> frame : frameTrack)
            {
                if (std::shared_ptr<WaveTrack> strongTrack = frame->track.lock())
                {
                    std::cout << "Track: " << strongTrack->GetName().ToStdString();
                    std::cout << " Frame: " << strongTrack->LongSamplesToTime(frame->start) << "-" << strongTrack->LongSamplesToTime(frame->start.as_size_t() + frame->length);
                    std::cout << " Silent: " << (frame->audioIsSilent() ? "YES" : "NO");
                    std::cout << " Modified: " << (frame->audioDidChange() ? "YES" : "NO") << std::endl;
                }
            }
            
            for (std::shared_ptr<IALAudioFrame> frame : frameTrack)
            {
                if (std::shared_ptr<WaveTrack> strongTrack = frame->track.lock())
                {
                    std::cout << "Track: " << strongTrack->GetName().ToStdString();
                    std::cout << " Samples: " << strongTrack->LongSamplesToTime(frame->start) << "-" << strongTrack->LongSamplesToTime(frame->start.as_size_t() + frame->length);
                    std::cout << " Silent: " << (frame->audioIsSilent() ? "YES" : "NO");
                    std::cout << " Modified: " << (frame->audioDidChange() ? "YES" : "NO") << std::endl;
                }
            }
        }
    }
}

std::vector<SampleBuffer> IALLabeler::fetchProjectAudio() {
    auto &project = this->context.project;
    TrackList &tracklist = TrackList::Get(project);
    
    TrackFactory &trackFactory = TrackFactory::Get(project);
    SampleBlockFactoryPtr sampleBlockFactory = trackFactory.GetSampleBlockFactory();
    
    for (Track *track : tracklist) {
        if (dynamic_cast<WaveTrack *>(track) != nullptr) {
            WaveTrack *waveTrack = (WaveTrack *)track;
            
            for (WaveClip *clip : waveTrack->GetAllClips()) {
                WaveClip copyClip(*clip, sampleBlockFactory, true);
                copyClip.ConvertToSampleFormat(floatSample);

                SampleBuffer buffer(copyClip.GetNumSamples().as_size_t(), floatSample);
                copyClip.GetSamples(buffer.ptr(), floatSample, copyClip.GetStartSample(), copyClip.GetNumSamples().as_size_t());
                
            }
        }
    }
    
    std::vector<SampleBuffer>ret;
    return ret;
}
