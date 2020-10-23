//
//  IALAudioFrame.cpp
//  Audacity
//
//  Created by Jack Wiig on 10/22/20.
//

#include "IALAudioFrame.hpp"

#include <math.h>
#include <algorithm>
#include <tgmath.h>

#include "../WaveTrack.h"
#include "../Track.h"
#include "../WaveClip.h"


#pragma mark AudioFrame


IALAudioFrame::IALAudioFrame(std::weak_ptr<WaveTrack> track, sampleCount start, size_t length, bool checkAudio)
    : track(track), start(start), length(length), label(""), currentHash(arc4random())
{
    if (checkAudio)
    {
        this->audioDidChange();
    }
}

std::unique_ptr<SampleBuffer> IALAudioFrame::fetchAudio(sampleFormat format, int sampleRate)
{
    // If the track is still alive
    if (std::shared_ptr<WaveTrack> strongTrack = track.lock())
    {
        // Get the audio into the return buffer
        auto buffer = std::make_unique<SampleBuffer>(SampleBuffer(length, floatSample));
        
        // Fetch Relevant Information
        AudacityProject *project = strongTrack->GetOwner()->GetOwner();
        SampleBlockFactoryPtr sbFactory = TrackFactory::Get(*project).GetSampleBlockFactory();
        sampleFormat originalFormat = strongTrack->GetSampleFormat();

        // Create a waveclip that will be used to convert the samples
        WaveClip conversionClip(sbFactory, originalFormat, strongTrack->GetRate(), strongTrack->GetWaveColorIndex());
        
        // Copy the samples into a buffer
        size_t sourceLength = sourceAudioLength();
        SampleBuffer transferBuffer(sourceLength, originalFormat);
                
        strongTrack->Get(transferBuffer.ptr(), originalFormat, start, sourceLength);
        
        // Copy into the clip and transform audio
        conversionClip.Append(transferBuffer.ptr(), originalFormat, length);
        
        // Pad to 1s
        if (length > sourceLength)
        {
            // This is where we could change the behavior of the last sample
            conversionClip.AppendSilence(length - sourceLength, 0);
        }
        
        if (strongTrack->GetSampleFormat() != format)
        {
            conversionClip.ConvertToSampleFormat(format);
        }
        
        conversionClip.Resample(sampleRate);
        
        // Copy one last time into a buffer.
        conversionClip.GetSamples(buffer->ptr(), format, conversionClip.GetStartSample(), conversionClip.GetNumSamples().as_size_t());
        
        return buffer;
    }
    
    return nullptr;
}


bool IALAudioFrame::audioIsSilent(float threshold)
{
    if (std::shared_ptr<WaveTrack> strongTrack = track.lock())
    {
        // Convert samples to times
        double startTime = strongTrack->LongSamplesToTime(start);
        double endTime = strongTrack->LongSamplesToTime(start.as_size_t() + sourceAudioLength());
        
        return 20*std::log10(strongTrack->GetRMS(startTime, endTime)) < threshold;
    }
    
    return false;
}

// This change detector works by hashing the audio sample at the start, middle, and end of an audio frame
// and then summing their totals. If the result matches the cached result, then there is a VERY high chance
// the audio did not change.
bool IALAudioFrame::audioDidChange()
{
    if (std::shared_ptr<WaveTrack> strongTrack = track.lock())
    {
        sampleFormat format = strongTrack->GetSampleFormat();
        size_t trueLength= sourceAudioLength();
        
        SampleBuffer startBuffer(1, format);
        strongTrack->Get(startBuffer.ptr(), format, start, 1);
        
        SampleBuffer middleBuffer(1, format);
        strongTrack->Get(middleBuffer.ptr(), format, (start.as_size_t() + trueLength) / 2, 1);
        
        SampleBuffer endBuffer(1, format);
        strongTrack->Get(endBuffer.ptr(), format, start.as_size_t() + trueLength, 1);
        
        float startSample = *(float *)startBuffer.ptr();
        float midSample = *(float *)middleBuffer.ptr();
        float endSample = *(float *)startBuffer.ptr();
        
        std::hash<float> float_hasher;
        size_t hash = float_hasher(startSample) + float_hasher(midSample) + float_hasher(endSample);
        
        if (hash == currentHash)
        {
            return false;
        }
        
        // Update hash
        currentHash = hash;
        return true;
    }
    
    return true;
}

size_t IALAudioFrame::sourceAudioLength()
{
    if (std::shared_ptr<WaveTrack> strongTrack = track.lock())
    {
        sampleCount lastSample = strongTrack->TimeToLongSamples(strongTrack->GetEndTime());
        
        // Should be length for all frames excluding the last.
        return std::min(start.as_size_t() + length, lastSample.as_size_t()) - start.as_size_t();
    }
    
    return 0;
}


#pragma mark AudioFrameTrack

IALAudioFrameTrack::IALAudioFrameTrack(std::weak_ptr<WaveTrack> track, bool checkAudio) : track(track)
{
    frames = std::vector<std::shared_ptr<IALAudioFrame>>(0);
    validateFrameTrack();
}

std::weak_ptr<IALAudioFrame> IALAudioFrameTrack::operator[](sampleCount startSample)
{
    validateFrameTrack();
    if (std::shared_ptr<WaveTrack> strongTrack = track.lock())
    {
        long idx = strongTrack->LongSamplesToTime(startSample);
        return std::weak_ptr<IALAudioFrame>(frames[idx]);
    }
    
    return std::weak_ptr<IALAudioFrame>();
}

void IALAudioFrameTrack::validateFrameTrack()
{
    if (std::shared_ptr<WaveTrack> strongTrack = track.lock())
    {
        size_t numFrames = ceil(strongTrack->GetEndTime());
        
        if (frames.size() != numFrames)
        {
            int lastIdx = std::max((int)frames.size() - 1, 0);
            
            frames.resize(numFrames);
            createFramesForRange(lastIdx, numFrames);
        }
    }
}

void IALAudioFrameTrack::createFramesForRange(int startIdx, int endIdx, bool checkAudio)
{
    if (std::shared_ptr<WaveTrack> strongTrack = track.lock())
    {
        int sr = strongTrack->GetRate();
        
        for (int currIdx = startIdx; currIdx < endIdx; currIdx++)
        {
            sampleCount startSample = strongTrack->TimeToLongSamples(currIdx);
            frames[currIdx] = (std::make_shared<IALAudioFrame>(IALAudioFrame(track, startSample, sr)));
        }
    }
}
