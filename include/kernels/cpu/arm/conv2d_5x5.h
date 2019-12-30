//
// Created by yang on 2019/11/19.
//

#ifndef TENSORSTACK_KERNELS_CPU_ARM_CONV2D_5X5_H
#define TENSORSTACK_KERNELS_CPU_ARM_CONV2D_5X5_H

#include "core/tensor.h"
#include "backend/common_structure.h"

namespace ts{
    namespace cpu {
        namespace arm{
            template<typename T>
            class TS_DEBUG_API Conv2d5x5 {
            public:
                static void conv2d_5x5_s1(const Tensor &x,
                                          const Padding2D &padding,
                                          float padding_value,
                                          const Tensor &w,
                                          Tensor &out);

                static void conv2d_5x5_s2(const Tensor &x,
                                          const Padding2D &padding,
                                          float padding_value,
                                          const Tensor &w,
                                          Tensor &out);

            };
        }//arm
    }//cpu
}//ts

#endif //TENSORSTACK_KERNELS_CPU_ARM_CONV2D_5X5_H
