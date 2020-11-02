
## audacity labeling 
this is the version jack made, forked from https://github.com/jhwiig/AudacityLabeling.git. For the python implementation of the labeler, see the branch `python-branch`

## building audacity (with labeler)
I've only tested this on MacOS Catalina (10.15.3)

this chunk is from mac/cmake_build.txt:

---
Clone wxWidgets 3.1.3:
```
git clone --recurse-submodules https://github.com/audacity/wxWidgets.git
cd wxWidgets
git checkout audacity-fixes-3.1.3
```
Change directory to the folder where wxWidgets was cloned and build it using:

sudo <path to Audacity source>/mac/scripts/build_wxwidgets

The config command should return "3.1.3" if the install was successful:
```
/usr/local/x86_64/bin/wx-config --version
```

Now that you have wxWidgets installed, edit your .bash_profile and add:
```
export WX_CONFIG=/usr/local/x86_64/bin/wx-config
```

---

now, build
```
mkdir build
cd build
cmake .. -G Xcode
xcodebuild -configuration Release
```

copy model file to the product's resources dir (ask for the model file for now)
```
cp ../ial-weights/ial-model.pt bin/Release/Audacity.app/Contents/Resources/ial-model.pt
```

__note__: if for some reason you get a linker error related to -ljack, build the project without jack:
```
cmake -use_pa_jack=off .. -G Xcode
```