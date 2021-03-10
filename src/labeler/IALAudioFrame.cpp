//
//  IALAudioFrame.cpp
//  Audacity
//
//  Created by Jack Wiig on 10/22/20.
//

#include "IALAudioFrame.hpp"

#include <math.h>
#include <algorithm>
#include "portaudio.h"
#include <tgmath.h>
#include <string>

#include "../WaveTrack.h"
#include "../Track.h"
#include "../WaveClip.h"
#include "../FileNames.h"
#include "IALModel.hpp"
#include "IALLabeler.hpp"

bool trackInTrackList(TrackList& tracklist, std::shared_ptr<LabelTrack> track){
    // Track *leader = *tracklist.FindLeader(track);
    TrackId id = track->GetId();

    for (Track *other: tracklist){
        TrackId otherId = other->GetId();

        if (otherId == id){
            return true;
        }
    }
    return false;
}

// #pragma mark AudioFrame - Public

IALAudioFrame::IALAudioFrame(IALAudioFrameCollection &collection, const sampleCount start, const size_t desiredLength)
    : collection(collection), start(start), desiredLength(desiredLength), cachedHash(arc4random())
{
}

//TODO: handle when locking the track fails. (i.e. track was destroyed)
AudacityLabel IALAudioFrame::getAudacityLabel(std::string labelstr){
    std::weak_ptr<WaveTrack> weakTrack = collection.getLeaderTrack();
    std::shared_ptr<WaveTrack> strongTrack = weakTrack.lock();

    if (std::shared_ptr<WaveTrack> strongTrack = weakTrack.lock())
    {
        float sR = float(collection.trackSampleRate());
        float startSample = float(start.as_double());
        float lenSample = float(sourceLength(*strongTrack));
        float startTime = startSample / sR;
        float endTime = (startSample + lenSample)/sR;

        return AudacityLabel(startTime, endTime, labelstr);
    } else{
        return AudacityLabel(0, 0, "error");
    }
    
}

// label the current audio frame.
AudacityLabel IALAudioFrame::label()
{
    if (!audioDidChange())
    {
        return cachedLabel;
    }
    
    if (audioIsSilent())
    {
        cachedLabel = getAudacityLabel("silence");
        return cachedLabel;
    }
    
    torch::Tensor modelInput = downmixedAudio();
    std::vector<std::string> predictions = collection.classifier.predictInstruments(modelInput, 0.3);

    cachedLabel = getAudacityLabel(predictions[0]);
    return cachedLabel;
}

// #pragma mark AudioFrame - Private

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
        cachedHash = size_t(newHash);
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

//TODO: test with  non-float input audio
torch::Tensor IALAudioFrame::downmixedAudio(sampleFormat format, int sampleRate)
{
    // auto buffer = std::make_unique<SampleBuffer>(desiredLength, format);
    size_t numChannels = collection.numChannels();
    
    torch::Tensor samples = torch::empty({1, static_cast<long long>(numChannels), static_cast<long long>(desiredLength)});

    // // TEST PLAYBACK
    // PaStream *stream;
    // PaError err;
    // err = Pa_Initialize();
    // if (err != paNoError) { raise(1); }
    // PaStreamParameters outputParameters;
    // outputParameters.device = Pa_GetDefaultOutputDevice();
    // outputParameters.sampleFormat = paFloat32;
    // outputParameters.channelCount = 1;
    // outputParameters.hostApiSpecificStreamInfo = NULL;
    // outputParameters.suggestedLatency = Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
    // //

    collection.iterateChannels([&](WaveTrack &channel, size_t idx, bool *stop)
    {
        sampleFormat originalFormat = channel.GetSampleFormat();
        double originalSampleRate = channel.GetRate();
        size_t actualLength = sourceLength(channel);
        
        // SampleBuffer channelBuffer = SampleBuffer(desiredLength, format);
        AudacityProject *project = channel.GetOwner()->GetOwner();
        SampleBlockFactoryPtr sbFactory = WaveTrackFactory::Get(*project).GetSampleBlockFactory();

        // copy channel's samples into buffer
        SampleBuffer buffer(actualLength, originalFormat);
        // fill up the buffer
        channel.Get(buffer.ptr(), originalFormat, start, actualLength);

        // make a separate clip where we will do the necessary conversions 
        WaveClip conversionClip(sbFactory, originalFormat, channel.GetRate(), channel.GetWaveColorIndex());

        // fill the clip with our buffer
        conversionClip.Append(buffer.ptr(), originalFormat, desiredLength);
        conversionClip.Flush();
        // conversionClip.SetSamples(buffer.ptr(), originalFormat, start, actualLength);

        // do the conversions
        conversionClip.ConvertToSampleFormat(floatSample);
        conversionClip.Resample(sampleRate);
        // SampleBuffer buffer(conversionClip.GetNumSamples().as_size_t(), floatSample);
        conversionClip.GetSamples(buffer.ptr(), floatSample, conversionClip.GetStartSample(), conversionClip.GetNumSamples().as_size_t());
        
        torch::Tensor channelTensor = torch::from_blob(buffer.ptr(),
                                                       desiredLength,
                                                       torch::TensorOptions().dtype(torch::kFloat32));
        
        samples.index_put_({0, torch::indexing::TensorIndex(static_cast<int64_t>(idx)), torch::indexing::Slice()}, channelTensor);

        // // TEST PLAYBACK
        // err = Pa_OpenStream(&stream, NULL, &outputParameters, sampleRate, paFramesPerBufferUnspecified, paNoFlag, NULL, NULL);
        // if (err != paNoError) { exit(1); }
        // if (stream) {
        //     err = Pa_StartStream( stream );
        //     if( err != paNoError ) { exit(1); }
        //     err = Pa_WriteStream(stream, buffer.ptr(), conversionClip.GetNumSamples().as_size_t());
        //     if (err != paNoError) { exit(1); }
        //     printf("Waiting for playback to finish.\n");
        //     while( ( err = Pa_IsStreamActive( stream ) ) == 1 ) { Pa_Sleep(100); }
        //     if( err < 0 ) { exit(1); }
        //     err = Pa_CloseStream( stream );
        //     if( err != paNoError ) { exit(1); }
        // }
        // //
    });
    
    // Downmix, then shape appropriately for the model.
    torch::Tensor monoAudio = collection.classifier.downmix(samples);
    return collection.classifier.padAndReshape(monoAudio);
}

// #pragma mark Collection - Public

// constructor. 
// each audacity track should have a framecollection, which should have a label track for itself.
IALAudioFrameCollection::IALAudioFrameCollection(IALModel &classifier, std::weak_ptr<WaveTrack> channel)
    : classifier(classifier)
{
    if (std::shared_ptr<WaveTrack> strongChannel = channel.lock())
    {
        leaderTrackId = strongChannel->GetId();
        labelTrack = std::make_shared<LabelTrack>();
    }
}

// adds a channel to the frame collection, only if it belongs to the same leader as the rest of the collection. 
// question: does the id of a channel equal the id of its leader track? that seems to be the assumption here
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

// count how many channels are in the collection
size_t IALAudioFrameCollection::numChannels()
{
    size_t count = 0;
    
    iterateChannels([&](WaveTrack &channel, size_t idx, bool *stop)
    {
        count+=1;
    });
    
    return count;
}

std::weak_ptr<WaveTrack> IALAudioFrameCollection::getLeaderTrack()
{
    return channels[0];
}

// #pragma mark Collection - Private

// iterate through channels
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

// check if the provided channel is already in the collection. 
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
        // NOTE: this assumes that the time window for a frame collection is 1 second
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

void IALAudioFrameCollection::labelAllFrames(const AudacityProject &project){
    // grab the tracklist
    TrackList &tracklist = TrackList::Get(const_cast<AudacityProject&>(project));

    std::vector<AudacityLabel> predictions;
    for (auto &frame : audioFrames){
        AudacityLabel prediction = frame.label();
        predictions.emplace_back(prediction);
    }

    // make sure to call this before coalescing to get accurate results. 
    std::string trackName = mostCommonLabel(predictions);
    setTrackTitle(trackName);

    predictions = coalesceLabels(predictions);

    // TODO: find a better way to create label tracks than this
    if (!predictions.empty()){
        wxString labelFileName = wxFileName(FileNames::DataDir(), trackName + ".txt").GetFullPath();
        wxTextFile labelFile(labelFileName);
        
        // In the event of a crash, the file might still be there. If so, clear it out and get it ready for reuse. Otherwise, create a new one.
        if (labelFile.Exists()) {
            labelFile.Clear();
        }
        else {
            labelFile.Create();
        }
        labelFile.Open();
        
        // Write each timestamp to file
        for (auto &label : predictions) {
            labelFile.AddLine(wxString(label.toStdString()));
        }

        labelTrack->SetName(wxString(trackName));
        labelTrack->Import(labelFile);

        if (!trackInTrackList(tracklist, labelTrack)) {
            tracklist.Add(labelTrack);
        }

        labelFile.Close();
        wxRemove(labelFileName);
    }

}

//TODO: handle when locking the track fails. (i.e. track was destroyed)
void IALAudioFrameCollection::setTrackTitle(const std::string &trackTitle){
    std::weak_ptr<WaveTrack> weakTrack = getLeaderTrack();
    std::shared_ptr<WaveTrack> strongTrack = weakTrack.lock();

    if (std::shared_ptr<WaveTrack> strongTrack = weakTrack.lock())
    {
        strongTrack->SetName(wxString(trackTitle));
    }
 
}

bool compareLabelCounts(std::pair<std::string, int> a, std::pair<std::string, int> b){
    return a.second < b.second;
}

std::string IALAudioFrameCollection::mostCommonLabel(const std::vector<AudacityLabel> &labels){
    // we should get the most common label in a given set of UNCOALESCED predictions
    // if the most common label is silence, use the second most common label
    // if there is no other common label, simply return silence
    // if the labels are empty, return silence

    std::string mostFreqLabel;
    if (labels.empty())
    {
        mostFreqLabel = "silence";
        return mostFreqLabel;
    }
    else
    {
        // make a dict with key label and value count
        std::map<std::string, int> counter;

        for (auto label: labels){
            // if we haven't seen the label b4, initialize to 0 
            if (counter.find(label.label) == counter.end()){
                counter[label.label] = 0;
            }
            // increment counter
            counter[label.label] += 1;
        }

        // sort map by value by putting it into a vector
        std::vector<std::pair<std::string, int>> sortedCounter;
        std::map<std::string, int>::iterator it;
        for (it=counter.begin(); it!=counter.end(); it++){
            sortedCounter.push_back(std::make_pair(it->first, it->second));
        }

        std::sort(sortedCounter.rbegin(), sortedCounter.rend(), compareLabelCounts);

        mostFreqLabel = sortedCounter[0].first;

        // if the most common label is silence, just grab the second most common label
        if (mostFreqLabel == "silence" && sortedCounter.size() > 1){
            mostFreqLabel = sortedCounter[1].first;
        }
    }
    return mostFreqLabel;
}
std::vector<AudacityLabel> IALAudioFrameCollection::coalesceLabels(const std::vector<AudacityLabel> &labels) {
    std::vector<AudacityLabel> coalescedLabels;
    
    // if labels are empty, don't return 
    if (labels.size() == 0) {
        return coalescedLabels;
    }
    
    int startIdx = 0;
    for (int i = 0; i < labels.size(); i++) {
        
        // Iterate over a range until a label differs from the previous label AND
        // previous label end time is same as the current label's start time

        // if the current label isn't the same as our base label, 
        // OR the previous label does not at the same time as the current label starts
        if (labels[i].label != labels[startIdx].label
            || labels[i - 1].end != labels[i].start) {
            coalescedLabels.push_back(AudacityLabel(labels[startIdx].start,
                                                    labels[i - 1].end,
                                                    labels[startIdx].label));
            startIdx = i;
        }
    }
    
    coalescedLabels.push_back(AudacityLabel(labels[startIdx].start,
                                            labels[labels.size() - 1].end,
                                            labels[startIdx].label));
    
    return coalescedLabels;
}