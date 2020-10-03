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

from labeler import preprocessors, audio_utils
import openl3


def embed(audio, sr, model):
    audio = audio_utils.downmix(audio)
    audio = audio_utils.zero_pad(audio, length=sr)

    embedding = model(audio, sr)

    return embedding

def embed_audio_file(path_to_audio, path_to_output, model):
    audio, sr = torchaudio.load(path_to_audio)
    audio = audio.detach().numpy()

    embedding = embed(audio, sr, model)

    pd.DataFrame(embedding).to_json(path_to_output)

def embed_audio_files(list_of_in_paths, list_of_out_paths, model):
    for in_path, out_path in zip(list_of_in_paths, list_of_out_paths):
        embed_audio_file(in_path, out_path, model)



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

    embed_audio_files(path_to_audio, path_to_output, model)

    
        



