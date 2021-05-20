#ifndef TENNIS_ENV_VARS_H
#define TENNIS_ENV_VARS_H

#include <utils/platform.h>
#include <string>

#if TS_PLATFORM_OS_WINDOWS
    #include <Windows.h>
#elif TS_PLATFORM_OS_LINUX || TS_PLATFORM_OS_ANDROID
    #include <cstdlib>
#endif

namespace ts {
    std::string getEnvironmentVariable(const std::string& envName);

}

#endif //TENNIS_ENV_VARS_H
