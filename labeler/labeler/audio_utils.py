import numpy as np
import librosa
import torchaudio

def resample(audio, old_sr, new_sr): 
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    return torchaudio.transforms.Resample(old_sr, new_sr)(audio)

def downmix(audio, keepdim = False):
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
   
    
def zero_pad(audio, length):
    """
    zero pad audio left and right to match length
    audio must be shape (N,) (mono)
    """
    assert audio.ndim == 1
    if len(audio) < length:
        pad_length = length - len(audio)
        pad_right = int(np.floor(pad_length/2))
        pad_left = int(np.ceil(pad_length/2))

        audio = np.pad(audio, (pad_left, pad_right), 'constant', constant_values=(0, 0))
    
    assert len(audio) >= length
    
    return audio
    