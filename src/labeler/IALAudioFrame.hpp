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
class IALAudioFrameCollection;

/**
 @brief A lightweight representation of a defined frame of audio in a specific WaveTrack.
 @discussion The goal of this class is to keep track of a region that serves as input to the label prediction model. This class can make observations about
 the source frame that allow it to determine if it is necessary to recompute the label. It also will transform the audio into the desired format for the input model when fetched.
 */
class IALAudioFrame
{
public:
    /**
     @brief The reference wave tracks that this audio frame is sourced from.
     @discussion This vector will either contain one (in the case of mono) or two (in the case of stereo) WaveTracks.
     When passing through the model, these are downmixed to mono.
     */
    std::weak_ptr<IALAudioFrameCollection> collection;
    
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
     @brief The cached label from the last time the model processed the audio.
     */
    std::string label;
    
    /**
     @brief The constructor for an audio frame, that establishes the location of the frame
     @param track The reference track
     @param start The starting sample in the reference track
     @param desiredLength the length of the frame in terms of samples in reference to the original track
     @param checkAudio (optional) a boolean indicating whether to find an initial hash value for didChange
     @returns an instance of an IALAudioFrame
     */
    IALAudioFrame(std::weak_ptr<IALAudioFrameCollection> collection, const sampleCount start, size_t desiredLength, bool checkAudio=false);
    
    /**
     @brief The true length of the frame. This method is calculated by comparing the source frame's duration to our desired length.
     @discussion This value is useful in determining where to apply the label. It also prevents accidentally reading past the malloc'ed region of memory.
     @returns the true length of the source frame in terms of samples.
     */
    size_t sourceLength(const std::weak_ptr<WaveTrack> &channel);
    
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
    
    
    std::string labelAudio(sampleFormat format=floatSample, int sampleRate=48000);
    
    /**
     @brief Returns a pointer to a buffer of audio from the reference track that is desiredLength samples long in the specified sample format and sample rate.
     @param format (optional) the bit representation of the samples of the audio to return. By default, 32-bit float is used
     @param sampleRate (optional) the number of samples that occur in a single second of audio, in Hertz. By default, 48kHz is used.
     @returns a pointer to a buffer of audio.
     @discussion This method returns a pointer to a buffer so that the move constructor of the pointer is used and not the copy constructor of the SampleBuffer.
     If the copy constructor is used, the data will be freed. The samplebuffer contains desiredLength samples so that it returns the fixed size the instantiator expects
     when creating the audio frame, even if the source audio is not of the proper length. Audio is resampled and sample format is changed if necessary before copying
     to the buffer.
     */
    std::unique_ptr<SampleBuffer> fetchAudio(sampleFormat format=floatSample, int sampleRate=48000);
    
private:
    /**
     @brief The hash value stored to represent the current state of the audio frame.
     */
    size_t cachedHash;
    
    std::unique_ptr<SampleBuffer> monoAudio(sampleFormat format=floatSample, int sampleRate=48000);
};

/**
 @brief A class that defines a collection of audio frames belonging to a single track.
 */
class IALAudioFrameCollection : std::enable_shared_from_this<IALAudioFrameCollection>
{
public:
    /**
     @brief The main WaveTrack associated with an audio track in Audacity.
     */
    const std::weak_ptr<WaveTrack> primaryTrack;
    
    /**
     @brief The initializer of an audio frame collection that takes a reference track.
     @param track The reference wave track that the collection will create frames for.
     @param checkAudio (optional) a boolean indicating whether to initialize the hashed state of a frame.
     @returns an instance of IALAudioFrameCollection
     */
    IALAudioFrameCollection(std::weak_ptr<WaveTrack> primaryTrack, bool checkAudio=false);
    
    /**
     @brief Ensures the correct number of frames are allocated for the reference wavetrack
     */
    void validateFrameCollection();
    
    
    const std::vector<std::weak_ptr<WaveTrack>> channels();
    void addChannel(std::weak_ptr<WaveTrack> channel);
    void removeChannel(std::weak_ptr<WaveTrack> channel);
    bool containsChannel(std::weak_ptr<WaveTrack> channel);
    
    std::vector<std::string> collectionLabels();
    
    // Iterator definition
    using audioFrames = std::vector<std::shared_ptr<IALAudioFrame>>;
    using iterator = typename audioFrames::iterator;
    using const_iterator = typename audioFrames::const_iterator;
    iterator begin() { return frames.begin(); }
    iterator end() { return frames.end(); }
    const_iterator cbegin() const { return frames.cbegin(); }
    const_iterator cend() const { return frames.cend(); }
    
private:
    /**
     @brief A collection of pointers to audio frames, which may or may not be valid.
     */
    std::vector<std::shared_ptr<IALAudioFrame>> frames;
    
    /**
     @brief The collection of tracks belonging to a single Audacity audio track. Typically this means a left and right channel in a stereo file.
     */
    std::vector<std::weak_ptr<WaveTrack>> channelVector;
    
    /**
     @brief Creates frames for the specified indices of the frames vector.
     @param startIdx the first frame to create (inclusive).
     @param endIdx the last frame to create (exclusive).
     @param checkAudio (optional) whether to properly initialize the hash value (default to false)
     */
    void createFramesForRange(int startIdx, int endIdx, bool checkAudio=false);
};



class IALAudioFrameCollection2 : std::enable_shared_from_this<IALAudioFrameCollection2>
{
public:
    
    IALAudioFrameCollection2(std::weak_ptr<WaveTrack> track);
    
    std::vector<std::weak_ptr<WaveTrack>> channels();
    bool addChannel(std::weak_ptr<WaveTrack> track);
    bool removeChannel(std::weak_ptr<WaveTrack> track);
    
    
    
private:
    class IALAudioFrame
    {
    public:
        const sampleCount start;
        const size_t desiredLength;
        
        std::string label();
        
        IALAudioFrame(const IALAudioFrameCollection2 &collection, const sampleCount start, size_t desiredLength);
        
    private:
        const IALAudioFrameCollection2 &collection;
        
        bool audioDidChange();
        bool audioIsSilent(float threshold=-80);
        bool downmixedAudio(sampleFormat format=floatSample, int sampleRate=48000);
    };
    
    std::vector<std::weak_ptr<WaveTrack>> channelVector;
    std::vector<IALAudioFrame> audioFrames;
    
    void handleDeletedTrack();
    void extendFramesToLength(size_t length);
};

#endif /* IALInputAudioFrame_hpp */
