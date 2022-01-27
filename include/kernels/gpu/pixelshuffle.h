//
// Created by sen on 2022/1/25.
//

#ifndef TENNIS_KERNELS_GPU_PIXELSHUFFLE_H
#define TENNIS_KERNELS_GPU_PIXELSHUFFLE_H

#include "backend/base/base_pixelshuffle.h"
#include "operator_on_gpu.h"

namespace ts {
    namespace gpu {
        class PixelShuffle : public OperatorOnGPU<base::PixelShuffle> {
        public:
            using self = PixelShuffle;
            using supper = OperatorOnGPU<base::PixelShuffle>;

            void pixel_shuffle(const Tensor& x, Tensor& out, int scale_factor, std::string mode) override;
        };
    }
}

#endif //TENNIS_KERNELS_GPU_PIXELSHUFFLE_H
