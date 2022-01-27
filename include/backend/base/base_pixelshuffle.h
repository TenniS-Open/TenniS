//
// Created by sen on 2022/1/25.
//

#ifndef TENNIS_BASE_PIXELSHUFFLE_H
#define TENNIS_BASE_PIXELSHUFFLE_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class PixelShuffle : public OperatorOnDevice {
        public:
            using self = PixelShuffle;
            using supper = OperatorOnDevice;

            PixelShuffle();
            void init() override;
            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;
            int run(Stack &stack) override;

            virtual void pixel_shuffle(const Tensor& x, Tensor& out, int upscale_factor, std::string mode) = 0;

        private:
            int m_upscale_factor = 1;
            std::string m_mode = "DCR";
        };
    }
}

#endif //TENNIS_BASE_PIXELSHUFFLE_H
