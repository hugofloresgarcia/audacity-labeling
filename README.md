
## audacity labeling 
this is the version jack made, forked from https://github.com/jhwiig/AudacityLabeling.git. For the python implementation of the labeler, see the branch `python-branch`

## building audacity (with labeler)
I've added jack's dependencies (torch and essentia) to the CMake build. Building should (hopefully) be easier now. 

I've only tested this on MacOS Catalina (10.15.3)

first, you need to build essentia from source. 
```
git clone git@github.com:mtg/essentia.git
cd essentia
./waf configure --with-tensorflow --with-cpptests --with-examples
./waf
./waf install
```
if this didn't work, you may need to install some additional libraries from brew. try https://essentia.upf.edu/installing.html.  

now, build audacity

```
mkdir build
cd build
cmake .. -G Xcode
xcodebuild -configuration Debug
```