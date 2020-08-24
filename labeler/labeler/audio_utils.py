import numpy
import librosa

def downmix(audio):
    # downmix if needed
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    return audio

def split_on_silence(audio, top_db=80):
    """
    split audio on silence using librosa
    returns:
        split_audio (list): list of np.arrays with split audio
        intervals (np.ndarray):  intervals[i] == 
                    (start_i, end_i) are the start and end time 
                    (in samples) of non-silent interval i.
    """
    intervals = librosa.effects.split(audio, 
                                    top_db=top_db, 
                                    frame_length=2048, 
                                    hop_length=512)
    
    split_audio = [audio[i[0]:i[1]] for i in intervals]

    return split_audio, intervals

    
