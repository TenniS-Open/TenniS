//
// Created by seeta on 2018/7/19.
//

#ifndef TENSORSTACK_KERNELS_COMMON_MATH_H
#define TENSORSTACK_KERNELS_COMMON_MATH_H

#include <cfloat>

namespace ts {
    inline bool near(double value1, double value2) {
        return (value1 > value2 ? value1 - value2 : value2 - value1) < DBL_EPSILON;
    }

    inline bool near(float value1, float value2) {
        return (value1 > value2 ? value1 - value2 : value2 - value1) < FLT_EPSILON;
    }
}


#endif //TENSORSTACK_KERNELS_COMMON_MATH_H
