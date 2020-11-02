mkdir build
cd build
cmake .. -G Xcode
xcodebuild -configuration Release
cp ../weights/bigpapa-mdb-epoch21.pt bin/Release/Audacity.app/Contents/Resources/bigpapa-mdb-epoch21.pt