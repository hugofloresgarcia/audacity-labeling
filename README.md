## audacity labeler

## installing the labeler (python 3.7)

cd into our labeler folder:  
`cd labeler`

(optional) make your very own venv  
`python3 -m venv venv` then `source venv/bin/activate`

install pip requirements  
you should really only need:
```
torch
torchaudio
pytorch_lightning
numpy
librosa
```

### get the model weights (from me)

make sure to put the weight file  on /labeler/model-weights

---

## building audacity (with python labeler)
I've only tested this on MacOS Catalina (10.15.3)

---

### this chunk is from mac/cmake_build.txt:

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

---

install zero-mq library using brew

*** note: *** zero-mq is used to build the client-server model that bridges python and C++. 
we shouldn't need this once we load the model in C++ using torchscript instead

`brew install zeromq`

---
now, build audacity

```
cd <path to Audacity source>
mkdir build
cd build
cmake .. -G Xcode
xcodebuild -configuration Debug
```

Audacity should now be in /build/bin/Debug/Audacity.app

---

## using with Audacity
cd into the labeler folder and run the server app:  
`python3 server.py`

now, you will be able to import labeled audio in audacity using  
File-->Import-->Labeled Audio
