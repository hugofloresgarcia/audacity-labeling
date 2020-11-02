
## audacity labeling 
this is the version jack made, forked from https://github.com/jhwiig/AudacityLabeling.git. For the python implementation of the labeler, see the branch `python-branch`

## building audacity (with labeler)
I've added jack's dependencies (torch and essentia) to the CMake build. Building should (hopefully) be easier now. 

I've only tested this on MacOS Catalina (10.15.3)

now, build audacity

```
mkdir build
cd build
cmake .. -G Xcode
xcodebuild -configuration Debug
```

__note__: if for some reason you get a linker error related to -ljack, build the project without jack:
```
cmake -use_pa_jack=off .. -G Xcode
```