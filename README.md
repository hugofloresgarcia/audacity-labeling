## audacity labeling 

## building audacity (with labeler)
I've only tested this on MacOS Catalina (10.15.3)

this chunk is from mac/cmake_build.txt:

---
### wxWidgets
Clone wxWidgets 3.1.3:
```
git clone --recurse-submodules https://github.com/audacity/wxWidgets.git
cd wxWidgets
git checkout audacity-fixes-3.1.3
```
Change directory to the folder where wxWidgets was cloned and build it using:

```
sudo <path to Audacity source>/mac/scripts/build_wxwidgets
```

The config command should return "3.1.3" if the install was successful:
```
/usr/local/x86_64/bin/wx-config --version
```

Now that you have wxWidgets installed, edit your .bash_profile and add:
```
export WX_CONFIG=/usr/local/x86_64/bin/wx-config
```
---
### libtorch

cd into the labeler source and download libtorch
```
cd <path to Audacity source>/src/labeler
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.7.0.zip
unzip libtorch-macos-1.7.0.zip
```
yee haw. 

---

now, build
```
mkdir build
cd build
cmake .. -G Xcode
xcodebuild -configuration Release
```

__note__: if for some reason you get a linker error related to -ljack, build the project without jack:
```
cmake -use_pa_jack=off .. -G Xcode
```
__note__: another issue we've run into is the local (Audacity) version of libsndfile breaking. If this happens to you, install libsndfile using brew, then configure using the system version:
```
cmake -USE_SNDFILE system .. -G Xcode
```

copy model file to the product's resources dir (ask for the model file for now)
AND the labels file. make sure to copy to the appropriate dir (Release or Debug)
```
cd .. # cd back to project root
cp ./ial-weights/medleydb/ial-model.pt build/bin/Release/Audacity.app/Contents/Resources/ial-model.pt
cp ./ial-weights/medleydb/ial-instruments.txt build/bin/Release/Audacity.app/Contents/Resources/ial-instruments.txt
```
