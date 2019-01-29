//
// Created by kier on 2018/7/19.
//

#ifndef TENSORSTACK_KERNELS_COMMON_MATH_H
#define TENSORSTACK_KERNELS_COMMON_MATH_H

#include <cfloat>
#include <cmath>
#include <cstdlib>

namespace ts {
    inline bool near(double value1, double value2) {
        return (value1 > value2 ? value1 - value2 : value2 - value1) < DBL_EPSILON;
    }

    inline bool near(float value1, float value2) {
        return (value1 > value2 ? value1 - value2 : value2 - value1) < FLT_EPSILON;
    }

    template <typename T>
    inline T abs(T value) {
        return T(std::abs(value));
    }

    template <>
    inline float abs(float value) {
        return std::fabsf(value);
    }

    template <>
    inline double abs(double value) {
        return std::fabs(value);
    }
}


#endif //TENSORSTACK_KERNELS_COMMON_MATH_H
