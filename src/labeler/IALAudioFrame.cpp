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


#pragma mark AudioFrame - Public

IALAudioFrame::IALAudioFrame(std::weak_ptr<IALAudioFrameCollection> collection, sampleCount start, size_t desiredLength, bool checkAudio)
    : collection(collection), start(start), desiredLength(desiredLength), label(""), cachedHash(arc4random())
{
    if (checkAudio)
    {
        audioDidChange();
    }
}

std::string IALAudioFrame::labelAudio(sampleFormat format, int sampleRate)
{
    std::unique_ptr<SampleBuffer> audio = monoAudio(format, sampleRate);

    //  load model
    IALModel model(/* modelPath: */ wxFileName(FileNames::ResourcesDir(), wxT("ial-model.pt")).GetFullPath().ToStdString(),
            /* instrumentListPath*/ wxFileName(FileNames::ResourcesDir(), wxT("ial-instruments.txt")).GetFullPath().ToStdString());

    // convert buffer to tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor audioTensor = torch::from_blob(audio->ptr(), desiredLength, options);

    // reshape into (batch, channels, sample)
    audioTensor = model.padAndReshape(audioTensor);

    // get back class labels from model
    std::vector<std::string> predictions = model.predictInstruments(audioTensor, /*confidenceThreshold: */0.3);
    for (const auto &e : predictions) std::cout << e << "\n";

    label = predictions[0];
    return predictions[0];
}


bool IALAudioFrame::audioIsSilent(float threshold)
{
    bool silent = true;

    for (std::weak_ptr<WaveTrack> channel : collection.lock()->channels())
    {
        if (std::shared_ptr<WaveTrack> strongChannel = channel.lock())
        {
            // Convert samples to times
            double startTime = strongChannel->LongSamplesToTime(start);
            double endTime = strongChannel->LongSamplesToTime(start.as_size_t() + sourceLength(channel));
            
            // Use short-circuiting to skip checks once one fails
            silent = silent && 20*std::log10(strongChannel->GetRMS(startTime, endTime)) < threshold;
        }

        else
        {
            // Handle invalid track?
            silent = false;
        }
    }
    
    if (silent)
    {
        label = "Silence";
    }
    
    return silent;
}

// This change detector works by hashing the audio sample at the start, middle, and end of an audio frame
// and then summing their totals. If the result matches the cached result, then there is a VERY high chance
// the audio did not change.
bool IALAudioFrame::audioDidChange()
{
    float sampleTotal = 0.0f;
    
    for (std::weak_ptr<WaveTrack> channel : collection.lock()->channels())
    {
        if (std::shared_ptr<WaveTrack> strongChannel = channel.lock())
        {
            sampleFormat format = strongChannel->GetSampleFormat();
            size_t actualLength = sourceLength(channel);
            
            SampleBuffer startBuffer(1, format);
            strongChannel->Get(startBuffer.ptr(), format, start, 1);
            
            SampleBuffer middleBuffer(1, format);
            strongChannel->Get(middleBuffer.ptr(), format, (start.as_size_t() + actualLength) / 2, 1);
            
            SampleBuffer endBuffer(1, format);
            strongChannel->Get(endBuffer.ptr(), format, start.as_size_t() + actualLength, 1);
            
            float startSample = *(float *)startBuffer.ptr();
            float midSample = *(float *)middleBuffer.ptr();
            float endSample = *(float *)startBuffer.ptr();
            
            sampleTotal += startSample + midSample + endSample;
        }
        
        else
        {
            // Handle invalid track?
        }
    }
    
    std::hash<float> float_hasher;
    size_t newHash = float_hasher(sampleTotal);
    
    if (cachedHash != newHash)
    {
        cachedHash = newHash;
        return true;
    }
    
    return false;
}


size_t IALAudioFrame::sourceLength(const std::weak_ptr<WaveTrack> &channel)
{
    if (std::shared_ptr<WaveTrack> strongChannel = channel.lock())
    {
        sampleCount lastSample = strongChannel->TimeToLongSamples(strongChannel->GetEndTime());
        
        // Should be length for all frames excluding the last.
        return std::min(start.as_size_t() + desiredLength, lastSample.as_size_t()) - start.as_size_t();
    }
    
    return 0;
}


#pragma mark AudioFrame - Private


std::unique_ptr<SampleBuffer> IALAudioFrame::monoAudio(sampleFormat format, int sampleRate)
{
    auto buffer = std::make_unique<SampleBuffer>(desiredLength, format);
    size_t numChannels = collection.lock()->channels().size();
    
    for (std::weak_ptr<WaveTrack> channel : collection.lock()->channels())
    {
        if (std::shared_ptr<WaveTrack> strongChannel = channel.lock())
        {
            sampleFormat originalFormat = strongChannel->GetSampleFormat();
            double originalSampleRate = strongChannel->GetRate();
            size_t actualLength = sourceLength(channel);
            
            SampleBuffer iterableBuffer = SampleBuffer(desiredLength, format);
            
            // Only modify the track if it needs to be changed
            if (originalFormat != format || originalSampleRate != sampleRate || actualLength != desiredLength)
            {
                // Fetch Project Information
                AudacityProject *project = strongChannel->GetOwner()->GetOwner();
                SampleBlockFactoryPtr sbFactory = TrackFactory::Get(*project).GetSampleBlockFactory();

                // Create a waveclip that will be used to convert the samples
                WaveClip conversionClip(sbFactory, originalFormat, strongChannel->GetRate(), strongChannel->GetWaveColorIndex());

                // Copy the samples into a buffer
                SampleBuffer transferBuffer(actualLength, originalFormat);

                strongChannel->Get(transferBuffer.ptr(), originalFormat, start, actualLength);
                
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
                conversionClip.GetSamples(iterableBuffer.ptr(), format, conversionClip.GetStartSample(), conversionClip.GetNumSamples().as_size_t());
            }
            else
            {
                strongChannel->Get(iterableBuffer.ptr(), format, start, desiredLength);
            }
            
            // Iterate through buffer to add to top-level
            for (size_t sampleIdx = 0; sampleIdx < desiredLength; sampleIdx += 1)
            {
                // Grab the sample from the channel
                float channelSample = *((float *)iterableBuffer.ptr() + sampleIdx);
            
                // Add it to the aggregate channel
                *((float *)buffer->ptr() + sampleIdx) += channelSample;
            }

        }
        
        else
        {
            numChannels -= 1;
            // Handle missing / deallocated channel
        }
    }
    
    // Average down
    if (numChannels > 0)
    {
        for (size_t sampleIdx = 0; sampleIdx < desiredLength; sampleIdx += 1)
        {
            *((float *)buffer->ptr() + sampleIdx) /= numChannels;
        }
    }
    
    else
    {
        return nullptr;
    }
    
    return buffer;
}



#pragma mark AudioFrameCollection

IALAudioFrameCollection::IALAudioFrameCollection(std::weak_ptr<WaveTrack> primaryTrack, bool checkAudio) : primaryTrack(primaryTrack)
{
    // Create IALAudioFrame Vector
    frames = std::vector<std::shared_ptr<IALAudioFrame>>(0);
    
    // Create Channel Vector
    channelVector = std::vector<std::weak_ptr<WaveTrack>>(2);
    channelVector.push_back(primaryTrack);
    
    validateFrameCollection();
}

const std::vector<std::weak_ptr<WaveTrack>> IALAudioFrameCollection::channels()
{
    return channelVector;
}

void IALAudioFrameCollection::addChannel(std::weak_ptr<WaveTrack> channel)
{
    channelVector.push_back(channel);
}

void IALAudioFrameCollection::removeChannel(std::weak_ptr<WaveTrack> channel)
{
    size_t idx = 0;
    std::shared_ptr<WaveTrack> strongChannel = channel.lock();
    
    for (std::weak_ptr<WaveTrack> otherChannel : channelVector)
    {
        if (strongChannel == otherChannel.lock())
        {
            channelVector.erase(channelVector.begin()+idx);
            break;
        }
        
        idx += 1;
    }
}

void IALAudioFrameCollection::validateFrameCollection()
{
    size_t maxNumFrames = 0;
    
    for (std::weak_ptr<WaveTrack> channel : channelVector)
    {
        if (std::shared_ptr<WaveTrack> strongChannel = channel.lock())
        {
            size_t numFrames = ceil(strongChannel->GetEndTime());
            maxNumFrames = std::max(maxNumFrames, numFrames);
        }
    }
    
    if (frames.size() != maxNumFrames)
    {
        size_t lastIdx = std::max((int)frames.size() - 1, 0);
        
        frames.resize(maxNumFrames);
        createFramesForRange(lastIdx, maxNumFrames);
    }
}

void IALAudioFrameCollection::createFramesForRange(int startIdx, int endIdx, bool checkAudio)
{
    if (std::shared_ptr<WaveTrack> strongPrimaryTrack = primaryTrack.lock())
    {
        int sr = strongPrimaryTrack->GetRate();
        
        auto weakSelf = std::weak_ptr<IALAudioFrameCollection>(shared_from_this());
        
        for (int currIdx = startIdx; currIdx < endIdx; currIdx++)
        {
            sampleCount startSample = strongPrimaryTrack->TimeToLongSamples(currIdx);
            frames[currIdx] = (std::make_shared<IALAudioFrame>(IALAudioFrame(weakSelf, startSample, sr)));
        }
    }
}

bool IALAudioFrameCollection::containsChannel(std::weak_ptr<WaveTrack> channel)
{
    for (std::weak_ptr<WaveTrack> existingChannel : channelVector)
    {
        if (existingChannel.lock() == channel.lock())
        {
            return true;
        }
    }
    
    return false;
}

std::vector<std::string> IALAudioFrameCollection::collectionLabels()
{
    std::vector<std::string> labels;
    
    std::string lastLabel = frames[0]->label;
    size_t startIdx = 0;
    
    for (size_t idx = 1; idx < frames.size(); idx++)
    {
        std::string frameLabel = frames[0]->label;
        
        // Write labels
        if (frameLabel != lastLabel)
        {
            std::string newLine = lastLabel + "\t" + std::to_string(startIdx) + "\t" + std::to_string(idx) + "\n";
            labels.push_back(newLine);
            
            startIdx = idx;
            lastLabel = frameLabel;
        }
    }
    
    return labels;
}
