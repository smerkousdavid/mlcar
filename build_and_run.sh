#!/bin/bash

echo "Building..."
CWD=$(pwd)
cd ~/ML/mlcar/build
make -j4

if [ $? -ne 0 ]; then
	cd $CWD
	echo "Failed to compile! Not continuing..."
	exit 1
fi

cd $CWD

echo "Done!"

~/ML/mlcar/push_binary.sh
~/ML/mlcar/run_binary.sh
