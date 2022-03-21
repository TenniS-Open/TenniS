#!/usr/bin/env bash

# ARCH: armv7 armv7s arm64 i386 x86_64

export TARGET=9 
export ENABLE_BITCODE=0
export PLATFORM=OS

export ARCH="arm64"
export TAG=xnnpack

HOME=$(cd `dirname $0`; pwd)
PROJECT="$HOME/../.."

# git checkout $TAG
cmake "$PROJECT" \
"$@" \
-DCMAKE_TOOLCHAIN_FILE="$PROJECT/toolchain/iOS.cmake" \
-DCMAKE_BUILD_TYPE=Release \
-DTS_USE_OPENMP=OFF \
-DTS_USE_SIMD=ON \
-DTS_USE_CBLAS=ON \
-DTS_ON_ARM=ON \
-DTS_USE_NEON=ON \
-DIOS_DEPLOYMENT_TARGET=$TARGET \
-DIOS_PLATFORM=$PLATFORM \
-DENABLE_BITCODE=$ENABLE_BITCODE \
-DIOS_ARCH="$ARCH"

