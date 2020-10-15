mkdir build
cd build
cmake .. -G Xcode
xcodebuild -configuration Release
cp ../weights/tunedopenl3_philharmonia_torchscript.pt bin/Release/Audacity.app/Contents/Resources/tunedopenl3_philharmonia_torchscript.pt