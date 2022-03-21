#!/usr/bin/env bash

HOME=$(cd `dirname $0`; pwd)
PROJECT="$HOME/../.."

cmake "$PROJECT" \
"$@" \
-DCMAKE_BUILD_TYPE=Release \
-DTS_DYNAMIC_INSTRUCTION=ON 

