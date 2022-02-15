#!/usr/bin/bash

export BUILD_DIR=build.linux.aarch64
export BUILD_TYPE=Release
export PLATFORM_TARGET=aarch64

export PLATFORM=x64
export INSTALL_DIR=$(cd "$(dirname "$0")"; pwd)/../../build

HOME=$(cd `dirname $0`; pwd)

cd $HOME

mkdir "$BUILD_DIR"

cd "$BUILD_DIR"


cmake "$HOME/.." \
-DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
-DCONFIGURATION="$BUILD_TYPE" \
-DPLATFORM="$PLATFORM_TARGET" \
-DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
-DTS_USE_OPENMP=OFF \
-DTS_USE_SIMD=OFF \
-DTS_ON_HASWELL=OFF \
-DTS_DYNAMIC_INSTRUCTION=ON \
-DTS_ON_ARM=ON

make VERBOSE=1

make install
