#!/usr/bin/env bash

# setting
SCRIPTS=(
    "android-arm64-v8a.sh"
    "android-armeabi-v7a.sh"
    "android-x86_64.sh"
    "android-x86.sh"
)
BRANCHS=(master master master master)

export GIT=git
export CMAKE=cmake

# check args
if [ $# -lt 1 ]; then
    echo "Usage: $0 install"
    exit 1
fi

# make path absolute
mkdir -p $1
INSTALL=$(cd "$1"; pwd)

# setup vars

PWD=`pwd`
HOME=$(cd `dirname $0`; pwd)
PROJECT="$HOME/../.."

SCRIPT_ROOT="$PROJECT/script/cmake"

N=${#SCRIPTS[@]}
for ((i=0; i<"$N"; i++)) do
    S=${SCRIPTS[$i]}
    B=${BRANCHS[$i]}

    # sub dir name
    T=${S%%.*}

    # sub dir path
    P="$PWD/$T"    
    mkdir -p "$P"

    echo "========[ Building $S"
    # checkout branch
    if [ -n "$B" ]; then
        pushd "$PROJECT" > /dev/null
        echo "[INFO] git checkout $B"
        $GIT checkout $B
        if [ $? -ne 0 ]; then
            echo "[ERROR] can not checkout branch to $B"
            exit $?
        fi
        popd > /dev/null
    fi

    # do cmake
    echo "[INFO] cmake ..."
    pushd "$P" > /dev/null
    $SHELL "${SCRIPT_ROOT}/$S" -DCMAKE_INSTALL_PREFIX="$INSTALL"
    if [ $? -ne 0 ]; then
        echo "[ERROR] cmake build porjcet filed in $S"
        exit $?
    fi

    # do make
    echo "[INFO] building ..."
    $CMAKE --build . --target install -- -j $THREADS
    if [ $? -ne 0 ]; then
        echo "[ERROR] compile project failed for $S"
        exit $?
    fi
    popd > /dev/null
done

echo "[SUCCEED] $INSTALL"

