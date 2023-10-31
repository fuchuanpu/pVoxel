#!/usr/bin/env bash

BASE_NAME=$(basename $(pwd))

readonly BASE_NAME

if [ $BASE_NAME != "pVoxel" ] && [ $BASE_NAME != "pvoxel" ]; then
    echo "This script should be executed in the root dir of pVoxel."
    exit -1
fi

sudo apt-get update -y
sudo apt-get install build-essential cmake ninja-build unzip -y

sudo apt install libgflags-dev -y
sudo apt install libboost-all-dev -y
sudo apt install libmlpack-dev mlpack-bin libarmadillo-dev -y

if [ ! -f "json.hpp" ]; then
    wget https://github.com/nlohmann/json/releases/download/v3.10.5/json.hpp
fi

function test_and_create_dir() {
    if [ $# != 1 ]; then
        echo "Invalid arguments."
        return -1;
    fi
    if [ ! -d "$1" ]; then
        mkdir "$1"
    fi
}

test_and_create_dir ./log