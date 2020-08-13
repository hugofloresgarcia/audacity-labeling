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

clone our two preprocessors to <path to audacity>/labeler/labeler/
`cd  <path to audacity>/labeler/labeler/preprocessors`
`git clone git@github.com:hugofloresgarcia/openl3.git`  
`git clone git@github.com:harritaylor/torchvggish.git`  

install openl3

`pip3 install ./openl3 `

## building audacity (with python labeler)
I've only tested this on MacOS Catalina (10.15.3)

#### this chunk is from mac/cmake_build.txt

Clone wxWidgets 3.1.3:

`git clone --recurse-submodules https://github.com/audacity/wxWidgets.git`  
`cd wxWidgets`  
`git checkout audacity-fixes-3.1.3`

Change directory to the folder where wxWidgets was cloned and build it using:

`sudo <path to Audacity source>/mac/scripts/build_wxwidgets`

The config command should return "3.1.3" if the install was successful:  
`/usr/local/x86_64/bin/wx-config --version`

Now that you have wxWidgets installed, edit your .bash_profile and add:  
`export WX_CONFIG=/usr/local/x86_64/bin/wx-config`

cool

install zero-mq library using brew

`brew install zeromq`

now, build audacity

```
cd <path to Audacity source>
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
note that the model must be a pytorch model and must be able to load using torch.load()
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

