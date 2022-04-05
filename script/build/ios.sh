#!/usr/bin/env bash

# setting
SCRIPTS=(
    "ios-arm64.sh"
    "ios-armv7.sh"
    "ios-x86_64.sh"
)
BRANCHS=(master master master)

THREADS=$(nproc 2>/dev/null || sysctl -n hw.ncpu)

export GIT=git
export CMAKE=cmake
export PYTHON=python3

# check args
if [ $# -lt 1 ]; then
    echo "Usage: $0 install [tmp]"
    exit 1
fi

TMP=""
if [ $# -gt 1 ]; then
    TMP="$2"
fi

# make path absolute
mkdir -p $1
INSTALL=$(cd "$1"; pwd)

# setup vars
PWD=`pwd`
HOME=$(cd `dirname $0`; pwd)
PROJECT="$HOME/../.."

SCRIPT_ROOT="$PROJECT/script/cmake"

# setup tmp

if [ -z "$TMP" ]; then
    PPATH=$(cd "$PROJECT"; pwd)
    PNAME=${PPATH##*/}
    TMP="/tmp/$PNAME"
fi

rm -rf "$TMP"
mkdir -p "$TMP"

echo "[INFO] Using $THREADS thread(s)"
echo "[INFO] Framworks temporary folder: $TMP"

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
    $SHELL "${SCRIPT_ROOT}/$S" -DCMAKE_INSTALL_PREFIX="$TMP/$T"
    if [ $? -ne 0 ]; then
        echo "[ERROR] cmake build porjcet filed in $S"
        exit $?
    fi

    # do make
    echo "[INFO] building ..."
    rm -rf "$PROJECT/lib"
    mkdir -p "$PROJECT/lib"
    $CMAKE --build . --target install -- -j $THREADS
    if [ $? -ne 0 ]; then
        echo "[ERROR] compile project failed for $S"
        exit $?
    fi
    popd > /dev/null
done

PACK_SCRIPT="$PROJECT/script/framework/pack.py"

$PYTHON "${PACK_SCRIPT}" -s "$TMP" "$INSTALL"
if [ $? -ne 0 ]; then
    echo "[ERROR] pack framework failed in $TMP"
    exit $?
fi

echo "[SUCCEED] $INSTALL"

