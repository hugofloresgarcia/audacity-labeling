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
#include <functional>

#include <torch/script.h>

class sampleCount;
class WaveTrack;
class SampleBuffer;
class IALModel;
class TrackId;

class IALAudioFrameCollection;

/**
 @brief A lightweight representation of a defined frame of audio in a collection of single or multichannel tracks.
 @discussion The goal of this class is to keep track of a region that serves as input to the label prediction model. This class can make observations about
 the source frame that allow it to determine if it is necessary to recompute the label. It also will transform the audio into the desired format for the input model when fetched.
 */
class IALAudioFrame
{
public:
    
    /**
     @brief The location in the original reference track that marks the beginning of the audio frame.
     */
    const sampleCount start;
    
    /**
     @brief The desired length of the audio frame in samples. The actual time duration of the frame is dependent on the sample rate of the track.
     @discussion The desired length is not necessarily the length of the source clip for this region. However, the audio fetched from this
     frame will be of desiredLength length. Currently, this is done by zero-padding a copy of the source clip to match desiredLength.
     */
    const size_t desiredLength;
    
    /**
     @brief Returns the automatic label assigned to this audio frame.
     @discussion This method will do a varying amount of work depending on how much has changed between the last calculation.
     If nothing changed, then the cachedLabel will be returned. If change is detected, a silence check will be run, and if the audio is not silent,
     the model will predict the label for the frame.
     */
    std::string label();
    
    /**
     @brief The constructor for an audio frame that establishes the location of the frame
     @param collection The collection instance that this frame belongs to
     @param start The starting sample in the reference track
     @param desiredLength the length of the frame in terms of samples in reference to the original track
     @returns an instance of an IALAudioFrame
     */
    IALAudioFrame(IALAudioFrameCollection &collection, const sampleCount start, const size_t desiredLength);
    
private:
    /**
     @brief The collection instance that this frame belongs to
     */
    IALAudioFrameCollection &collection;
    
    /**
     @brief The last computed hash of the audio frame, stored as a canary to check for audio changes
     */
    size_t cachedHash;
    
    /**
     @brief The last computed label of the audio frame, stored in case the label is requested before the audio changes.
     */
    std::string cachedLabel;
    
    /**
     @brief The length of the frame within the context of the track. Either desiredLength or the remainder of the track, whichever is shorter.
     */
    size_t sourceLength(WaveTrack &track);
    
    /**
     @brief Detects whether the source audio in this frame has changed from the last time it was checked.
     be detected, then the rest of the frame does not need to be loaded or passed into a model.
     @returns a boolean indicating change.
     @discussion This method picks the first, middle, and last sample in the frame, hashes them, and then sums them to produce one final
     result. If this number differs from the cached number, then the audio has changed for this region. The idea is to minimize how much audio
     needs to be fetch every pass.
     */
    bool audioDidChange();
    
    /**
     @brief Detects if the source audio frame is silent using RMS and converting to dBFS (deciBels Full-Scale)
     @param threshold (optional) the average energy the track needs to be above to be determined not silent.
     @returns a boolean indicating whether the region is silent or not.
     */
    bool audioIsSilent(float threshold=-80);
    
    /**
     @brief Returns a tensor of audio from the frame that is desiredLength samples long in the specified sample format and sample rate.
     @param format (optional) the bit representation of the samples of the audio to return. By default, 32-bit float is used
     @param sampleRate (optional) the number of samples that occur in a single second of audio, in Hertz. By default, 48kHz is used.
     @returns a torch tensor containing downmixed (averaged) audio of the channels
     @discussion The tensor contains desiredLength samples so that it returns the fixed size the instantiator expects when creating
     the audio frame, even if the source audio is not of the proper length. Audio is resampled and sample format is changed if necessary
     before copying to the buffer.
     */
    torch::Tensor downmixedAudio(sampleFormat format=floatSample, int sampleRate=48000);
};


/**
 @brief A class that manages the creation and updating of labels on a single or multichannel track.
 @discussion This class is the interface for a single track, from the perspective of an end user. It will handle the creation
 and updating of a label track associated with a single or multichannel track by using a frame-wise representation and managing
 updates when the frames are no longer valid. This class will also handle its own deletion by making calls to its parent, the IALLabeler.
 */
class IALAudioFrameCollection : std::enable_shared_from_this<IALAudioFrameCollection>
{
public:
    IALModel &classifier;
    
    IALAudioFrameCollection(IALModel &classifier, std::weak_ptr<WaveTrack> channel);
        
    size_t numChannels();
    void iterateChannels(std::function<void(WaveTrack &channel, size_t idx, bool *stop)> loopBlock);
    bool addChannel(std::weak_ptr<WaveTrack> channel);
    std::vector<IALAudioFrame> audioFrames;
    std::shared_ptr<LabelTrack> labelTrack;
    
private:
    std::vector<std::weak_ptr<WaveTrack>> channels;
    TrackId leaderTrackId;
    
    size_t trackSampleRate();
    
    bool containsChannel(std::weak_ptr<WaveTrack> channel);
    void handleDeletedChannel(size_t deletedChannelIdx);
    void updateCollectionLength();
};

#endif /* IALInputAudioFrame_hpp */
