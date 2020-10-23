//
//  IALAudioFrame.hpp
//  Audacity
//
//  Created by Jack Wiig on 10/22/20.
//

#ifndef IALAudioFrame_hpp
#define IALAudioFrame_hpp

#include <stdio.h>
#include <memory>
#include <vector>

class sampleCount;
class WaveTrack;
class SampleBuffer;

class IALAudioFrame
{
public:
    /// Reference Track
    const std::weak_ptr<WaveTrack> track;
    
    /// Starting location in the track, with respect to the original track's sample rate
    const sampleCount start;
    
    /// Length of the frame, with respect to the original track's sample rate.
    const size_t length;
    
    /// Label for this time-stamp
    std::string label;
    
    IALAudioFrame(const std::weak_ptr<WaveTrack> track, const sampleCount start, size_t length, bool checkAudio=false);
    
    /// Returns whether the audio in the frame was modified
    bool audioDidChange();
    
    /// Returns whether the audio in the region is silent
    bool audioIsSilent(float threshold=-80);
        
    /// Returns a pointer to a buffer (to prevent accientally calling the destructor) that fetches the audio after converting it to 32-bit float and
    /// resampling to 48,000Hz
    std::unique_ptr<SampleBuffer> fetchAudio(sampleFormat format=floatSample, int sampleRate=48000);
    
private:
    /// A hash value used to approximate the state of an audio frame for later comparison (to detect changes).
    size_t currentHash;
    
    size_t sourceAudioLength();
};

class IALAudioFrameTrack
{
public:
    /// Reference Track
    const std::weak_ptr<WaveTrack> track;
    
    IALAudioFrameTrack(std::weak_ptr<WaveTrack> track, bool checkAudio=false);
    
    /// The audio frames are indexed by their starting time (which should be on the second)
    std::weak_ptr<IALAudioFrame> operator[](sampleCount);
    
    // Iterator definition
    using audioFrames = std::vector<std::shared_ptr<IALAudioFrame>>;
    using iterator = typename audioFrames::iterator;
    using const_iterator = typename audioFrames::const_iterator;
    iterator begin() { return frames.begin(); }
    iterator end() { return frames.end(); }
    const_iterator cbegin() const { return frames.cbegin(); }
    const_iterator cend() const { return frames.cend(); }
    
private:
    std::vector<std::shared_ptr<IALAudioFrame>> frames;
    
    /// Recalculates the length of the frame track based on the wave track's length
    void validateFrameTrack();
    
    /// Creates samples for the range.
    void createFramesForRange(int startIdx, int endIdx, bool checkAudio=false);
};

#endif /* IALInputAudioFrame_hpp */
