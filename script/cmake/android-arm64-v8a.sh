#!/usr/bin/env bash

# ANDROID_ABI can be armeabi-v7a arm64-v8a x86 x86_64

export ANDROID_PLATFORM=19
export ANDROID_ABI="arm64-v8a"

# check if set ndk home
if [ -z ${ANDROID_NDK} ]; then
    echo "[ERROR] The environment variable ANDROID_NDK must be set."
    exit 1
elif [ ! -f "${ANDROID_NDK}/build/cmake/android.toolchain.cmake" ]; then
    echo "[ERROR] Can not access ${ANDROID_NDK}/build/cmake/android.toolchain.cmake"
    exit 2
fi

HOME=$(cd `dirname $0`; pwd)
PROJECT="$HOME/../.."

cmake "$PROJECT" \
"$@" \
-DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
-DCMAKE_BUILD_TYPE=Release \
-DTS_ON_ARM=ON \
-DTS_USE_NEON=ON \
-DTS_USE_FAST_MATH=ON \
-DPLATFORM="$ANDROID_ABI" \
-DANDROID_ABI="$ANDROID_ABI" \
-DANDROID_PLATFORM=android-$ANDROID_PLATFORM \
-DANDROID_STL=c++_static \
-DANDROID_ARM_NEON=ON

