//
// Created by kier on 2019/1/12.
//

#ifndef TENSORSTACK_BACKEND_COMMON_STRUCTURE_H
#define TENSORSTACK_BACKEND_COMMON_STRUCTURE_H

#include <cstdint>

namespace ts {
    template <typename T>
    class Aspect2D {
    public:
        using Dtype = T;

        Aspect2D() = default;
        Aspect2D(Dtype top, Dtype bottom, Dtype left, Dtype right)
                : top(top), bottom(bottom), left(left), right(right) {}

        Dtype top;
        Dtype bottom;
        Dtype left;
        Dtype right;
    };

    template <typename T>
    class Form2D {
    public:
        using Dtype = T;

        Form2D() = default;
        Form2D(Dtype height, Dtype width)
                : height(height), width(width) {}

        Dtype height;
        Dtype width;
    };

    using Padding2D = Aspect2D<int32_t>;

    using Stride2D = Form2D<int32_t>;

    using KSize2D = Form2D<int32_t>;

    using Dialations2D = Form2D<int32_t>;

    using Size2D = Form2D<int32_t>;
}

#endif //TENSORSTACK_BACKEND_COMMON_STRUCTURE_H
