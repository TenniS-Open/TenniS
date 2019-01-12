//
// Created by kier on 2019/1/12.
//

#ifndef TENSORSTACK_BACKEND_COMMON_FUNCTION_H
#define TENSORSTACK_BACKEND_COMMON_FUNCTION_H

#include "common_structure.h"

#include <algorithm>
#include <cmath>

namespace ts {
    inline Size2D pooling2d_forward(const Size2D &x, const Padding2D &padding, const KSize2D &ksize,
                                    const Stride2D &stride) {
        Size2D y;
        y.height = int(std::floor((x.height + padding.top + padding.bottom - ksize.height) / (float)stride.height + 1));
        y.width = int(std::floor((x.width + padding.left + padding.right - ksize.width) / (float)stride.width + 1));
        return y;
    }

    inline Size2D pooling2d_backward(const Size2D &y, const Padding2D &padding, const KSize2D &ksize,
                                     const Stride2D &stride) {
        Size2D x;
        x.height = (y.height - 1) * stride.height + ksize.height - padding.top - padding.bottom;
        x.width = (y.width - 1) * stride.width + ksize.width - padding.left - padding.right;
        return x;
    }
}


#endif //TENSORSTACK_BACKEND_COMMON_FUNCTION_H
