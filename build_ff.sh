FF_PATH=/usr/FlexFlow

cd $FF_PATH
rm -rf build
mkdir build
cd build
../config/config.linux
make -j64
