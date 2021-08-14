#ifndef TENNIS_ENV_VARS_H
#define TENNIS_ENV_VARS_H

#include <string>

namespace ts {
    std::string getEnvironmentVariable(const std::string& envName);

}

#endif //TENNIS_ENV_VARS_H
