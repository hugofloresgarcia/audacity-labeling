cd labeler
python3 -m venv venv-labeler
source venv-labeler/bin/activate
pip3 install -r requirements.txt


git clone git@github.com:hugofloresgarcia/openl3.git
git clone git@github.com:harritaylor/torchvggish.git  


pip3 install -e openl3 

cd ..
git clone --recurse-submodules https://github.com/audacity/wxWidgets.git
cd wxWidgets
git checkout audacity-fixes-3.1.3
cd ..
sudo mac/scripts/build_wxwidgets

/usr/local/x86_64/bin/wx-config --version
export WX_CONFIG=/usr/local/x86_64/bin/wx-config

mkdir build
cd build
cmake .. -G Xcode
xcodebuild -configuration Debug
