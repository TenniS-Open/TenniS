# GCC flags

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pessimizing-move")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_CREATE_SHARED_LIBRARY "${CMAKE_CXX_CREATE_SHARED_LIBRARY} -fPIC")
set(CMAKE_CXX_CREATE_SHARED_MODULE "${CMAKE_CXX_CREATE_SHARED_MODULE} -fPIC")
# set(CMAKE_CXX_CREATE_STATIC_LIBRARY "${CMAKE_CXX_CREATE_STATIC_LIBRARY}")

if (MINGW)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mxsave")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mxsave")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

if (TS_USE_FAST_MATH)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" PROCESSOR)
if ("${PROCESSOR}" MATCHES "aarch64" OR
    "${PROCESSOR}" MATCHES "arm64" OR
    "${PROCESSOR}" MATCHES "armv8")
    if (ANDROID)
        # This is legacy settings. Maybe had no usage, but kept.
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfloat-abi=hard")        # could be softfp as well
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfloat-abi=hard")    # could be softfp as well
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon-vfpv4")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-vfpv4")
    endif()
elseif ("${PROCESSOR}" MATCHES "armv7")
    if (ANDROID)
        # it seems that default android ndk cross compiler does not support hard
    else()
        # change toolchain to set float-abi, so that's it...
        # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfloat-abi=hard")
        # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfloat-abi=hard")
    endif ()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon-vfpv4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-vfpv4")
endif()

if (ANDROID)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-command-line-argument")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -frtti")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -funsafe-math-optimizations -ftree-vectorize")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funsafe-math-optimizations -ftree-vectorize")
    list(APPEND third_libraries log)
endif()

if ("${PLATFORM}" STREQUAL "x86")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
elseif (${PLATFORM} STREQUAL "x64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
endif()

if ("${CONFIGURATION}" STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g -ggdb")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -ggdb")
else()
    if (ANDROID)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()
