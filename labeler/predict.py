"""
predict.py
created by hugo flores garcia on 08/04/2020

"""
from labeler import classifiers
from labeler import preprocessors
from embed import embed

import numpy as np
import torch
import torchaudio
import pandas as pd
import argparse


def predict_from_audio_files(path_to_audio):
    assert isinstance(path_to_audio, list)

    embedding_name = 'vggish'
    classifier_name = '../mac/Resources/classifier.pt'

    # load model
    emb_model = preprocessors.get_model(embedding_name)
    classifier_model = classifiers.get_model('jack')

    predictions = []
    for path in path_to_audio:
        audio, sr = torchaudio.load(path)
        audio = audio.detach().numpy()

        emb, emb_ts = embed(audio, sr, emb_model)

        # now, make our env a torch tensor for classification
        pred, pred_ts = classifier_model.predict(emb, emb_ts)
        predictions.append((pred, pred_ts))
    
    return predictions

def label_audacity_track(prediction, ts, output_path):
    label_track = ''
    for idx, pred in enumerate(prediction):
        start = ts[idx]
        if (idx+1) == len(prediction):
            stop = ts[idx] + (ts[idx] - ts[idx-1])
        else: 
            stop = ts[idx+1]

        label_track += f'{start}\t{stop}\t{pred}\n'

    with open(output_path, 'w') as f:
        f.write(label_track)
        
    return label_track
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--paths_to_audio", type=str, nargs="+", 
                        default='/Users/hugoffg/Music/songs/bloom/bloom.mp3')
    parser.add_argument("-o", "--paths_to_output", type=str, nargs="+", 
                    default="/Users/hugoffg/Documents/lab/audacity-labeling/labeler/output/bloom.txt")

    args = parser.parse_args()

    paths_to_audio = args.paths_to_audio
    paths_to_output = args.paths_to_output

    if isinstance(paths_to_audio, str):
        paths_to_audio = [paths_to_audio]
    if isinstance(paths_to_output, str):
        paths_to_output = [paths_to_output]
    
    predictions = predict_from_audio_files(paths_to_audio)

    for path, prediction in zip(paths_to_output, predictions):
        pred, ts = prediction
        label_audacity_track(pred, ts, path)