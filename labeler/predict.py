"""
predict.py
created by hugo flores garcia on 08/04/2020

"""
from labeler import classifiers
from labeler import preprocessors
from labeler import audio_utils
from embed import embed

import numpy as np
import torch
import torchaudio
import pandas as pd
import argparse
import yaml


def predict_from_audio_files(path_to_audio, params = None):
    if params is None:
        preprocessor_name='vggish'
        preprocessor_kwargs = None
        classifier_name = 'jack'
        classifier_kwargs = None
    else:
        print(params)
        preprocessor_name = params['preprocessor']['name']
        preprocessor_kwargs = params['preprocessor']['preprocessor_kwargs'] \
            if 'preprocessor_kwargs' in params['preprocessor'] else None

        classifier_name = params['classifier']['name']
        classifier_kwargs = params['classifier']['classifier_kwargs'] \
            if 'classifier_kwargs' in params['classifier'] else None
    assert isinstance(path_to_audio, list) 

    # load model
    print(f'loading model {preprocessor_name} with kwargs {preprocessor_kwargs}')
    print(f'loading classifier {classifier_name} with kwargs {classifier_kwargs}')

    emb_model = preprocessors.get_model(preprocessor_name, preprocessor_kwargs)
    classifier_model = classifiers.get_model(classifier_name, classifier_kwargs)

    predictions = []
    for path in path_to_audio:
        print(f'about to predict labels for {path}...')
        audio, sr = torchaudio.load(path)
        audio = audio.detach().numpy()

        audio_list, intervals = audio_utils.split_on_silence(audio, top_db=80
        print('embedding audio...')
        emb, emb_ts = embed(audio, sr, emb_model)

        # now, make our env a torch tensor for classification
        print('classifying audio...')
        pred, pred_ts = classifier_model.predict(emb, emb_ts)
        pred_ts[-1] = len(audio[0])/sr
        predictions.append((pred, pred_ts))

        print(f'done!\n')
    return predictions

def label_audacity_track(prediction, ts, output_path):
    label_track = ''
    print(f'writing label track to {output_path}...')
    print(prediction)
    print(ts) 
    for idx, pred in enumerate(prediction):
        start = ts[idx]
        stop = ts[idx+1]

        label_track += f'{start}\t{stop}\t{pred}\n'

    with open(output_path, 'w') as f:
        f.write(label_track)
    
    return label_track
        

def predict_audacity_labels(
            paths_to_audio, 
            paths_to_output, 
            path_to_config="./labeler/labeler-config.yaml"):
    if isinstance(paths_to_audio, str):
        paths_to_audio = [paths_to_audio]
    if isinstance(paths_to_output, str):
        paths_to_output = [paths_to_output]

    config = yaml.load(path_to_config)
    print(config)
    config = {
        'preprocessor': {
            'name': 'openl3-mel256-6144-music',
            'model_kwargs': {
                'path_to_model': 'openl3-mel256-6144-music'
            }
        },
        'classifier': {
            'name': 'openl3_svm-linear_19-class',
            'classifier_kwargs': {
                'path_to_model': './classifiers/openl3_svm-linear_1s-chunks_no-weights',
                'classes': "saxophone,flute,guitar,contrabassoon,bass-clarinet,trombone,cello,oboe,bassoon,banjo,mandolin,tuba,viola,french-horn,english-horn,violin,double-bass,trumpet,clarinet".split(',')
            }
        }
    } 
    predictions = predict_from_audio_files(paths_to_audio, params=config)

    for path, prediction in zip(paths_to_output, predictions):
        pred, ts = prediction
        label_audacity_track(pred, ts, path)

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