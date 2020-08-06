

## building audacity (with labeler)
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