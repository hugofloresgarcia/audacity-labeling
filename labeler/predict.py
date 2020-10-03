"""
predict.py
created by hugo flores garcia on 08/04/2020

"""
from labeler import classifiers
from labeler import preprocessors
from labeler import audio_utils
from embed import embed
from labeler.tunedl3 import TunedOpenL3

import numpy as np
import torch
import torchaudio
import pandas as pd
import argparse
import yaml


def predict_from_audio_file(path_to_audio, preprocessor_name, classifier_name):
    print(f'loading model {preprocessor_name}')
    print(f'loading classifier {classifier_name}')

    # load embedding and classifier model
    model = TunedOpenL3.load_from_checkpoint('./model-weights/tunedl3.pt')

    print(f'about to predict labels for {path_to_audio}...')
    # load audio using torchaudio 
    audio, sr = torchaudio.load(path_to_audio)
    audio = audio.detach().numpy()

    # split on silence
    audio = audio_utils.downmix(audio)
    audio_list, intervals = audio_utils.split_on_silence(audio, top_db=80)
    intervals = intervals/sr

    padded_audio = []
    for aud in audio_list:
        padded_audio.append(audio_utils.zero_pad(aud, length=sr))

    padded_audio = np.stack(padded_audio, axis=0)
    
    print('classifying audio...')
    pred, pred_ts = model.predict(padded_audio, intervals)

    print(f'done!\n')
    return pred, pred_ts

def predict_from_audio_files(list_of_paths, preprocessor_name, classifier_name):
    predictions = []
    for path in list_of_paths:
        pred, pred_ts = predict_from_audio_file(path, preprocessor_name, classifier_name)
        predictions.append((pred, pred_ts))
    
    return predictions

def label_audacity_track(prediction, ts, output_path):
    label_track = ''
    print(f'writing label track to {output_path}...')
    print(prediction)
    print(ts) 

    # coalesce labels
    prediction_coalesced = []
    ts_coalesced = []
    for idx in range(len(prediction)):
        if idx == len(prediction):
            break
        pred, timestamp = prediction[idx], ts[idx]
        if (idx + 1) == len(ts): 
            pass
        else:
            start, end = timestamp
            future_start, future_end = ts[idx+1]

            current_pred = pred
            future_pred = prediction[idx+1]

            if np.isclose(end, future_start) and current_pred == future_pred:
                new_start = start
                new_end = future_end

                new_stamp = np.array([new_start, new_end])

                pred = pred
                timestamp = new_stamp

                prediction = np.delete(prediction, idx+1, axis=0)
                ts = np.delete(ts,idx+1, axis=0)

        
        prediction_coalesced.append(pred)
        ts_coalesced.append(timestamp)
    

    prediction = np.array(prediction_coalesced)
    ts = np.array(ts_coalesced)

    for idx, event in enumerate(zip(prediction, ts)):
        pred, timestamp = event
        start, stop = timestamp
        label_track += f'{start}\t{stop}\t{pred}\n'


    with open(output_path, 'w') as f:
        f.write(label_track)
    
    return label_track 

def write_audacity_labels(
            path_to_audio: str, 
            path_to_label: str, 
            preprocessor_name: str='openl3-mel256-6144-music', 
            classifier_name: str='openl3-svm-trim_silence'):

    prediction, ts = predict_from_audio_file(path_to_audio, preprocessor_name, classifier_name)
    label_audacity_track(prediction, ts, path_to_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--paths_to_audio", type=str, nargs="+", 
                        default='/Users/hugoffg/Music/songs/bloom/bloom.mp3')
    parser.add_argument("-o", "--paths_to_output", type=str, nargs="+", 
                        default="/Users/hugoffg/Documents/lab/audacity-labeling/labeler/output/bloom.txt")
    parser.add_argument("-c", "--config", type=str, nargs=1, 
                        default="./labeler/labeler-config.yaml")

    args = parser.parse_args()

    paths_to_audio = args.paths_to_audio
    paths_to_output = args.paths_to_output
    path_to_config = args.config

    predict_audacity_labels(paths_to_audio, paths_to_output, path_to_config)