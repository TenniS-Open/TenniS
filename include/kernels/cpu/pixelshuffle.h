//
// Created by sen on 2022/1/25.
//

#ifndef TENNIS_PIXELSHUFFLE_H
#define TENNIS_PIXELSHUFFLE_H

#include "backend/base/base_pixelshuffle.h"
#include "operator_on_cpu.h"

namespace ts {
    namespace cpu {
        class PixelShuffle : public OperatorOnCPU<base::PixelShuffle> {
        public:
            using self = PixelShuffle;
            using supper = OperatorOnCPU<base::PixelShuffle>;

            void pixel_shuffle(const Tensor& x, Tensor& out, int scale_factor, std::string mode) override;
        };
    }
}

#endif //TENNIS_PIXELSHUFFLE_H
