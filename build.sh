#!/usr/bin/env bash


BASE_NAME=$(basename $(pwd))

readonly BASE_NAME

if [ $BASE_NAME != "pVoxel" ] && [ $BASE_NAME != "pvoxel" ]; then
    echo "This script should be executed in the root dir of pVoxel."
    exit -1
fi

echo "Rebuild/Build pVoxel."

if [ -d "./build" ]; then
    echo "Old build dir is detected, remove it."
    rm -r ./build
fi

mkdir build && cd $_ && cmake -G Ninja .. && ninja  && cd ..
if [ $? == 0 ]; then
    echo "Rebuild finished."
else
    echo "Rebuild failed."
fi
