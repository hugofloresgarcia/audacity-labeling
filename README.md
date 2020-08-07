## audacity labeling!! python-branch

I'm trying to add the labeler as a small python server that creates label tracks for audacity tracks. 

The client-server code is written using the Zero-MQ library for both the C++ and python side. I know absolutely nothing about programming networks and sockets so if anybody has any suggestions I'd be happy to hear! :)

### advantages
- no more build times! build audacity once and change your models through python. 

## installing the labeler (python 3.7)

cd into our labeler folder:  
`cd labeler`

(optional) make your very own venv  
`python3 -m venv venv-labeler` then `source venv-labeler/bin/activate`

install pip requirements  
`pip3 install -r requirements.txt`

clone our two preprocessors  
`git clone git@github.com:hugofloresgarcia/openl3.git`  
`git clone git@github.com:harritaylor/torchvggish.git`  

install openl3

`pip3 install -e openl3 `

## building audacity (with python labeler)
I've only tested this on MacOS Catalina (10.15.3)

first, install ZeroMQ using brew.   
`brew install zeromq`

now, build audacity

```
mkdir build
cd build
cmake .. -G Xcode
xcodebuild -configuration Debug
```

Audacity should now be in /build/bin/Debug/Audacity.app

## using with Audacity
cd into the labeler folder and run the server app:  
`python3 server.py`

now, you will be able to import labeled audio in audacity using  
File-->Import-->Labeled Audio

## command line usage (labeler)

cd into our labeler folder:  
`cd labeler`

use `predict.py` to create individual label track files for a set for audio paths:  
`python3 predict.py -p song1.mp3 song2.wav song3.m4a -o song1.txt song2.txt song3.txt`  

make sure to provide proper paths for each input and output file.  
after the .txt files have been created, import them to audacity through File->Import->Labels... menu  

## chaging the preprocessor and classifier models
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

## loading a precompiled model
to label a precompiled model, format the labeler-config.yaml as follows:
```
preprocessor:
    name: custom
    preprocessor_kwargs: 
        path_to_model: /path/to/model # add the path to your .pt file here
        

classifier:
    name: custom
    classifier_kwargs: 
            path_to_model: /path/to/model # add the path to your .pt file here
```         

