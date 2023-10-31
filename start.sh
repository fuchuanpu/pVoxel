#!/usr/bin/env bash

BASE_NAME=$(basename $(pwd))

readonly BASE_NAME

if [ $BASE_NAME != "pVoxel" ] && [ $BASE_NAME != "pvoxel" ]; then
    echo "This script should be executed in the root dir of pVoxel."
    exit -1
fi

wget https://pvoxel-release.s3.cn-north-1.amazonaws.com.cn/data.zip
unzip data.zip
rm $_

chmod +x ./env.sh && $_ 
chmod +x ./build.sh && $_

cd build

./pVoxel -config ../config/N3IC.json
./pVoxel -config ../config/CICFlowMeter.json

cd ..

echo "Please find the results in ./log."
echo "Done."
