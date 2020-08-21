"""
embed.py
created by hugo flores garcia on 08/04/2020

embed an audio files using the models in preprocessors.py
"""

import numpy as np
import os
import pandas as pd
import argparse
import torchaudio

from labeler import preprocessors
import openl3


def embed(audio, sr, model):
    if isinstance(audio, list):
        batch=True
    elif isinstance(audio, np.ndarray):
        batch=False
        audio = [audio]
        sr = [sr]
    else:
        raise ValueError("audio must be an np array if or list (if batch)")
    
    embeddings = []
    for a, rate in zip(audio, sr):
        # pad with zeros if needed
        if a.shape[1] < rate:
            l = a.shape[1]
            z = np.zeros(rate - l)
            a = np.concatenate([a[0], z])
            a = np.expand_dims(a, 0)
        # downmix if neededa
        if a.ndim == 2:
            a = a.mean(axis=0)

        embedding = model(a, rate)
        embeddings.append(embedding)
    
    # remove from list if we didn't get a batch to begin with
    if not batch:
        embeddings = embeddings[0]

    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path_to_audio", type=str, nargs="+")
    parser.add_argument("-o", "--path_to_output", type=str, nargs="+")
    parser.add_argument("-m", "--model", type=str, default="openl3")

    args = parser.parse_args()

    path_to_audio = args.path_to_audio
    path_to_output = args.path_to_output
    model_name = args.model.lower()
    
    assert isinstance(path_to_audio, list)
    assert isinstance(path_to_output, list)
    assert len(path_to_audio) == len(path_to_output), "inputs and outputs must be the same"

    # load model
    model = preprocessors.get_model(model_name)
    embeddings = []
    for path in path_to_audio:
        audio, sr = torchaudio.load(path_to_audio)
        audio = audio.detach().numpy()

        embedding = embed(audio, sr, model)
        embeddings.append(embedding)
    
    for path, embedding in zip(path_to_output, embeddings):
        pd.DataFrame(embedding).to_json(path)

    
        



