//
// Created by yang on 2019/12/16.
//

#ifndef TENSORSTACK_KERNELS_CPU_ARM_CONV2D_3X3_V2_H
#define TENSORSTACK_KERNELS_CPU_ARM_CONV2D_3X3_V2_H


#include "core/tensor.h"
#include "backend/common_structure.h"

namespace ts {
namespace cpu {
namespace arm {

    template<typename T>
    class TS_DEBUG_API Conv2d3x3V2 {
    public:
        static void conv2d_3x3_s1(const Tensor &x,
                                  const Padding2D &padding,
                                  float padding_value,
                                  const Tensor &w,
                                  Tensor &out);
    };

} //arm
} //cpu
} //ts

#endif //TENSORSTACK_KERNELS_CPU_ARM_CONV2D_3X3_V2_H
