mkdir build
cd build
cmake .. -G Xcode
xcodebuild -configuration Release
cp ../mac/Resources/vggish.pb bin/Release/Audacity.app/Contents/Resources/vggish.pb
cp ../mac/Resources/classifier_instruments.txt bin/Release/Audacity.app/Contents/Resources/classifier_instruments.txt
cp ../mac/Resources/classifier.pt bin/Release/Audacity.app/Contents/Resources/classifier.pt