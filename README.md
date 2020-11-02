## audacity labeling 
For the python implementation of the labeler, see the branch `python-branch`

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
AND the labels fils
```
cp ../ial-weights/medleydb/ial-model.pt bin/Release/Audacity.app/Contents/Resources/ial-model.pt
cp ../ial-weights/medleydb/ial-instruments.txt bin/Release/Audacity.app/Contents/Resources/ial-instruments.txt
```

__note__: if for some reason you get a linker error related to -ljack, build the project without jack:
```
cmake -use_pa_jack=off .. -G Xcode
```