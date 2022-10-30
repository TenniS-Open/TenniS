//
// Created by Levalup on 2020/4/21.
//

#ifndef _INC_SEETA_AIP_PLATFORM_H
#define _INC_SEETA_AIP_PLATFORM_H

/**
 * Platform
 * including
 *  SEETA_AIP_OS_WINDOWS
 *  SEETA_AIP_OS_MAC
 *  SEETA_AIP_OS_LINUX
 *  SEETA_AIP_OS_IOS
 *  SEETA_AIP_OS_ANDROID
 *  SEETA_AIP_OS_UNIX
 */
 
#if defined(__ANDROID__)
#   define SEETA_AIP_OS_ANDROID 1
#   define SEETA_AIP_OS_WINDOWS 0
#   define SEETA_AIP_OS_MAC     0
#   define SEETA_AIP_OS_LINUX   1
#   define SEETA_AIP_OS_IOS     0
#   define SEETA_AIP_OS_UNIX    1
#elif defined(__WINDOWS__) || defined(_WIN32) || defined(WIN32) || defined(_WIN64) || \
    defined(WIN64) || defined(__WIN32__) || defined(__TOS_WIN__)
#   define SEETA_AIP_OS_ANDROID 0
#   define SEETA_AIP_OS_WINDOWS 1
#   define SEETA_AIP_OS_MAC     0
#   define SEETA_AIP_OS_LINUX   0
#   define SEETA_AIP_OS_IOS     0
#   define SEETA_AIP_OS_UNIX    0
#elif defined(__MACOSX) || defined(__MACOS_CLASSIC__) || defined(__APPLE__) || defined(__apple__)
#include "TargetConditionals.h"
#if TARGET_IPHONE_SIMULATOR || TARGET_OS_IPHONE
#   define SEETA_AIP_OS_ANDROID 0
#   define SEETA_AIP_OS_WINDOWS 0
#   define SEETA_AIP_OS_MAC     0
#   define SEETA_AIP_OS_LINUX   0
#   define SEETA_AIP_OS_IOS     1
#   define SEETA_AIP_OS_UNIX    0
#elif TARGET_OS_MAC
#   define SEETA_AIP_OS_ANDROID 0
#   define SEETA_AIP_OS_WINDOWS 0
#   define SEETA_AIP_OS_MAC     1
#   define SEETA_AIP_OS_LINUX   0
#   define SEETA_AIP_OS_IOS     0
#   define SEETA_AIP_OS_UNIX    1
#else
//#   error "Unknown Apple platform"
#   define SEETA_AIP_OS_ANDROID 0
#   define SEETA_AIP_OS_WINDOWS 0
#   define SEETA_AIP_OS_MAC     0
#   define SEETA_AIP_OS_LINUX   0
#   define SEETA_AIP_OS_IOS     0
#   define SEETA_AIP_OS_UNIX    0
#endif
#elif defined(__linux__) || defined(linux) || defined(__linux) || defined(__LINUX__) || \
    defined(LINUX) || defined(_LINUX)
#   define SEETA_AIP_OS_ANDROID 0
#   define SEETA_AIP_OS_WINDOWS 0
#   define SEETA_AIP_OS_MAC     0
#   define SEETA_AIP_OS_LINUX   1
#   define SEETA_AIP_OS_IOS     0
#   define SEETA_AIP_OS_UNIX    1
#elif defined(_UNIX) || defined(_unix) || defined(_UNIX_) || defined(_unix_)
#   define SEETA_AIP_OS_ANDROID 0
#   define SEETA_AIP_OS_WINDOWS 0
#   define SEETA_AIP_OS_MAC     0
#   define SEETA_AIP_OS_LINUX   0
#   define SEETA_AIP_OS_IOS     0
#   define SEETA_AIP_OS_UNIX    1
#else
//#   error Unknown OS
#   define SEETA_AIP_OS_ANDROID 0
#   define SEETA_AIP_OS_WINDOWS 0
#   define SEETA_AIP_OS_MAC     0
#   define SEETA_AIP_OS_LINUX   0
#   define SEETA_AIP_OS_IOS     0
#   define SEETA_AIP_OS_UNIX    0
#endif

#endif //_INC_SEETA_AIP_PLATFORM_H
