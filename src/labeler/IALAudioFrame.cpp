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
#include <string>

#include "../WaveTrack.h"
#include "../Track.h"
#include "../WaveClip.h"
#include "IALModel.hpp"
#include "IALLabeler.hpp"


#pragma mark AudioFrame - Public

IALAudioFrame::IALAudioFrame(IALAudioFrameCollection &collection, const sampleCount start, const size_t desiredLength)
    : collection(collection), start(start), desiredLength(desiredLength), cachedHash(arc4random())
{
}

std::string IALAudioFrame::label()
{
    if (!audioDidChange())
    {
        return cachedLabel;
    }
    
    if (audioIsSilent())
    {
        cachedLabel = "Silence";
        return cachedLabel;
    }
    
    torch::Tensor modelInput = downmixedAudio();
    std::vector<std::string> predictions = collection.classifier.predictInstruments(modelInput, 0.3);
    return predictions[0];
}

#pragma mark AudioFrame - Private

bool IALAudioFrame::audioIsSilent(float threshold)
{
    bool silent = true;
    
    collection.iterateChannels([&](WaveTrack &channel, size_t idx, bool *stop)
    {
        double startTime = channel.LongSamplesToTime(start);
        double endTime = channel.LongSamplesToTime(start.as_size_t() + sourceLength(channel));
        
        if (20 * std::log10(channel.GetRMS(startTime, endTime)) > threshold)
        {
            *stop = true;
            silent = false;
        }
    });
    
    return silent;
}

// This change detector works by hashing the audio sample at the start, middle, and end of an audio frame
// and then summing their totals. If the result matches the cached result, then there is a VERY high chance
// the audio did not change.
bool IALAudioFrame::audioDidChange()
{
    float sampleTotal = 0.0f;
    
    collection.iterateChannels([&](WaveTrack &channel, size_t idx, bool *stop)
    {
        sampleFormat format = channel.GetSampleFormat();
        size_t actualLength = sourceLength(channel);
        
        // Grab first, middle, and last sample from each track
        
        SampleBuffer startBuffer(1, format);
        channel.Get(startBuffer.ptr(), format, start, 1);
        
        SampleBuffer middleBuffer(1, format);
        channel.Get(middleBuffer.ptr(), format, (start.as_size_t() + actualLength) / 2, 1);
        
        SampleBuffer endBuffer(1, format);
        channel.Get(endBuffer.ptr(), format, start.as_size_t() + actualLength, 1);
        
        float startSample = *(float *)startBuffer.ptr();
        float midSample = *(float *)middleBuffer.ptr();
        float endSample = *(float *)startBuffer.ptr();
        
        sampleTotal += startSample + midSample + endSample;
    });
    
    std::hash<float> float_hasher;
    size_t newHash = float_hasher(sampleTotal);
    
    if (cachedHash != newHash)
    {
        cachedHash = newHash;
        return true;
    }
    
    return false;
}


size_t IALAudioFrame::sourceLength(WaveTrack &track)
{
    // Should be length for all frames excluding the last.
    sampleCount lastSample = track.TimeToLongSamples(track.GetEndTime());
    return std::min(start.as_size_t() + desiredLength, lastSample.as_size_t()) - start.as_size_t();
}


torch::Tensor IALAudioFrame::downmixedAudio(sampleFormat format, int sampleRate)
{
    auto buffer = std::make_unique<SampleBuffer>(desiredLength, format);
    size_t numChannels = collection.numChannels();
    
    torch::Tensor samples = torch::empty({1, static_cast<long long>(numChannels), static_cast<long long>(desiredLength)});
    
    collection.iterateChannels([&](WaveTrack &channel, size_t idx, bool *stop)
    {
        sampleFormat originalFormat = channel.GetSampleFormat();
        double originalSampleRate = channel.GetRate();
        size_t actualLength = sourceLength(channel);
        
        SampleBuffer channelBuffer = SampleBuffer(desiredLength, format);
        
        // Only modify the track if it needs to be changed
        if (originalFormat != format
            || originalSampleRate != sampleRate
            || actualLength != desiredLength)
        {
            // Fetch Project Information
            AudacityProject *project = channel.GetOwner()->GetOwner();
            SampleBlockFactoryPtr sbFactory = WaveTrackFactory::Get(*project).GetSampleBlockFactory();

            // Create a waveclip that will be used to convert the samples
            WaveClip conversionClip(sbFactory, originalFormat, channel.GetRate(), channel.GetWaveColorIndex());

            // Copy the samples into a buffer
            SampleBuffer transferBuffer(actualLength, originalFormat);

            channel.Get(transferBuffer.ptr(), originalFormat, start, actualLength);
            
            // Copy into the clip and transform audio
            conversionClip.Append(transferBuffer.ptr(), originalFormat, desiredLength);
            
            // CONVERSIONS: Length, Format, and Sample Rate
            
            if (desiredLength > actualLength)
            {
                // This is where we could change the behavior of the last sample
                conversionClip.AppendSilence(desiredLength - actualLength, 0);
            }
            
            if (originalFormat != format)
            {
                conversionClip.ConvertToSampleFormat(format);
            }
            
            if (originalSampleRate != sampleRate)
            {
                conversionClip.Resample(sampleRate);
            }
            
            // Copy one last time into a buffer.
            conversionClip.GetSamples(channelBuffer.ptr(), format, conversionClip.GetStartSample(), conversionClip.GetNumSamples().as_size_t());
        }
        else
        {
            channel.Get(channelBuffer.ptr(), format, start, desiredLength);
        }
        
        torch::Tensor channelTensor = torch::from_blob(channelBuffer.ptr(),
                                                       desiredLength,
                                                       torch::TensorOptions().dtype(torch::kFloat32));
        
        samples.index_put_({0, torch::indexing::TensorIndex(static_cast<int64_t>(idx)), torch::indexing::Slice()}, channelTensor);
    });
    
    // Downmix, then shape appropriately for the model.
    torch::Tensor monoAudio = collection.classifier.downmix(samples);
    return collection.classifier.padAndReshape(monoAudio);
}


#pragma mark Collection - Public



IALAudioFrameCollection::IALAudioFrameCollection(IALModel &classifier, std::weak_ptr<WaveTrack> channel)
    : classifier(classifier)
{
    if (std::shared_ptr<WaveTrack> strongChannel = channel.lock())
    {
        leaderTrackId = strongChannel->GetId();
        std::shared_ptr<TrackList> tracklist = strongChannel->GetOwner();
        TrackIter<Track> iter = tracklist->FindLeader(strongChannel.get());
//        std::shared_ptr<WaveTrack> primaryChannel = *(tracklist->FindLeader(strongChannel.get()));
    }
}


bool IALAudioFrameCollection::addChannel(std::weak_ptr<WaveTrack> channel)
{
    if (!containsChannel(channel))
    {
        if (std::shared_ptr<WaveTrack> strongChannel = channel.lock())
        {
            TrackId trackId = strongChannel->GetId();
            
            if (trackId == leaderTrackId)
            {
                channels.push_back(channel);
                updateCollectionLength();
                return true;
            }
        }
    }
    return false;
}


size_t IALAudioFrameCollection::numChannels()
{
    size_t count = 0;
    
    iterateChannels([&](WaveTrack &channel, size_t idx, bool *stop)
    {
        count+=1;
    });
    
    return count;
}


#pragma mark Collection - Private

void IALAudioFrameCollection::iterateChannels(std::function<void (WaveTrack &, size_t, bool *)> loopBlock)
{
    bool stopIteration = false;
    size_t currentIdx = 0;
    size_t iterIdx = 0;
    
    for (std::weak_ptr<WaveTrack> weakTrack : channels)
    {
        if (std::shared_ptr<WaveTrack> strongTrack = weakTrack.lock())
        {
            loopBlock(*strongTrack, currentIdx, &stopIteration);
            currentIdx += 1;
            
            if (stopIteration)
            {
                break;
            }
        }
        
        else
        {
            handleDeletedChannel(iterIdx);
        }
        
        iterIdx += 1;
    }
}


bool IALAudioFrameCollection::containsChannel(std::weak_ptr<WaveTrack> channel)
{
    bool contains = false;
    if (std::shared_ptr<WaveTrack> strongChannel = channel.lock())
    {
        iterateChannels([&](WaveTrack &track, size_t idx, bool *stop)
        {
            if (strongChannel->GetId() == track.GetId())
            {
                contains = true;
            }
        });
    }
    return contains;
}

void IALAudioFrameCollection::handleDeletedChannel(size_t deletedChannelIdx)
{
    size_t numChannels = channels.size();
    
    for (size_t idx = deletedChannelIdx; idx < numChannels; idx++)
    {
        if (channels[idx].expired())
        {
            channels.erase(channels.begin()+idx);
            numChannels -= 1;
        }
    }
    
    if (channels.size() == 0)
    {
        // delete frame
        
    }
    
    updateCollectionLength();
}

size_t IALAudioFrameCollection::trackSampleRate()
{
    size_t sampleRate;
    
    iterateChannels([&](WaveTrack &channel, size_t idx, bool *stop)
    {
        if (idx == 0)
        {
            sampleRate = channel.GetRate();
        }
        
        else if (sampleRate != channel.GetRate())
        {
            std::runtime_error("Differing Sample Rates in AudioFrame");
        }
    });
    
    return sampleRate;
}

void IALAudioFrameCollection::updateCollectionLength()
{
    size_t maxFrameCount = 0;
    
    iterateChannels([&](WaveTrack &channel, size_t idx, bool *stop)
    {
        size_t framesInChannel = ceil(channel.GetEndTime());
        maxFrameCount = std::max(maxFrameCount, framesInChannel);
    });
    
    if (audioFrames.size() != maxFrameCount)
    {
        size_t previousFrameCount = audioFrames.size();
    
        if (previousFrameCount < maxFrameCount)
        {
            size_t sampleRate = trackSampleRate();
            
            for (size_t frameIdx = previousFrameCount; frameIdx < maxFrameCount; frameIdx++)
            {
                sampleCount startSample = frameIdx * sampleRate;
                audioFrames.emplace_back(IALAudioFrame(*this, startSample, sampleRate));
            }
        }
        
        else
        {
            audioFrames.resize(maxFrameCount, IALAudioFrame(*this, sampleCount(0), 0));
        }
    }
}
