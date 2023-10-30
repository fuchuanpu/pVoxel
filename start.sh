#!/usr/bin/env bash

BASE_NAME=$(basename $(pwd))

readonly BASE_NAME

if [ $BASE_NAME != "pVoxel" ] && [ $BASE_NAME != "pvoxel" ]; then
    echo "This script should be executed in the root dir of pVoxel."
    exit -1
fi

./env.sh
./build.sh

cd build

./pVoxel -config ../config/N3IC.json
./pVoxel -config ../config/CICFlowMeter.json

cd ..

echo "Done."