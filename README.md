## audacity labeling!! python-branch

I'm trying to add the labeler as a small python server that creates label tracks for audacity tracks. 

I haven't implemented the client-server code yet, but the labeler is fully functional as a command line tool. 

### installing the labeler (python 3.7)

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
            path_to_model: /path/to/model # add the path to your .pt file here
```         

## building audacity (with cpp labeler)
I've only tested this on MacOS Catalina (10.15.3)

first, you need to build essentia from source.   
```
git clone git@github.com:mtg/essentia.git
cd essentia
./waf configure --with-tensorflow --with-cpptests --with-examples
./waf
./waf install
```

now, build audacity

```
mkdir build
cd build
cmake .. -G Xcode
xcodebuild -configuration Debug
```