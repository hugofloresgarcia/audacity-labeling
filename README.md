## audacity labeling!! python-branch

I'm trying to add the labeler as a small python server that creates label tracks for audacity tracks. 

I haven't implemented the client-server code yet, but the labeler is fully functional as a command line tool. 

### usage (labeler)

cd into our labeler folder:  
`cd labeler`

use `predict.py` to create individual label track files for a set for audio paths:  
`python3 predict.py -p song1.mp3 song2.wav song3.m4a -o song1.txt song2.txt song3.txt`  

make sure to provide proper paths for each input and output file.  
after the .txt files have been created, import them to audacity through File->Import->Labels... menu  

### chaging the preprocessor and classifier models
if you would like to change what preprocessor-classifier model is used, head to `labeler/labeler/labeler-config.yaml` and change the model using the following syntax:

***important***: make sure that the  preprocessor name you're referring to has been implemented in `labeler/classifiers.py` or `labeler/preprocessors.py`. to load a precompiled model, see the next section. 

if you are running an experiment and would like to provide your own custom config (instead of editing the main one) use the `--config` flag. 

`python3 predict.py -p song1.wav -o song2.txt --config /path/to/my_config.yaml`

```
preprocessor:
    name: openl3
    preprocessor_kwargs: # a set of keyworded arguments that will be passed to the preprocessor's __init__()
        hop_size: 1
        embedding_size: 512
        

classifier:
    name: my_classifier
    classifier_kwargs: 
```

### loading a precompiled model
to label a precompiled model, format the labeler-config.yaml as follows:
```
preprocessor:
    name: custom
    preprocessor_kwargs: 
        path_to_model: /path/to/model # add the path to your .pt file here
        

classifier:
    name: custom
    classifier_kwargs: 
            path_to_model: /path/to/model # add the path to your .pt file her
```         


[![Audacity](https://forum.audacityteam.org/styles/prosilver/theme/images/Audacity-logo_75px_trans_forum.png)](https://www.audacityteam.org) 
=========================

[**Audacity**](https://www.audacityteam.org) is an easy-to-use, multi-track audio editor and recorder for Windows, Mac OS X, GNU/Linux and other operating systems. Developed by a group of volunteers as open source.

- **Recording** from any real, or virtual audio device that is available to the host system.
- **Export / Import** a wide range of audio formats, extendible with FFmpeg.
- **High quality** using 32-bit float audio processing.
- **Plug-ins** Support for multiple audio plug-in formats, including VST, LV2, AU.
- **Macros** for chaining commands and batch processing.
- **Scripting** in Python, Perl, or any language that supports named pipes.
- **Nyquist** Very powerful built-in scripting language that may also be used to create plug-ins.
- **Editing** multi-track editing with sample accuracy and arbitrary sample rates.
- **Accessibility** for VI users.
- **Analysis and visualization** tools to analyze audio, or other signal data.

## Getting Started

For end users, the latest Windows and macOS release version of Audacity is available from the [Audacity website](https://www.audacityteam.org/download/).
Help with using Audacity is available from the [Audacity Forum](https://forum.audacityteam.org/).
Information for developers is available from the [Audacity Wiki](https://wiki.audacityteam.org/wiki/For_Developers).
