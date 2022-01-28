//
// Created by sen on 2022/1/25.
//

#ifndef TENNIS_KERNELS_XNNPACK_PIXELSHUFFLE_H
#define TENNIS_KERNELS_XNNPACK_PIXELSHUFFLE_H

#include "backend/base/base_pixelshuffle.h"
#include "kernels/cpu/operator_on_cpu.h"
#include "pthreadpool.h"
#include "kernels/xnnpack/xnnpack.h"


namespace ts {
    namespace xnn {
        class PixelShuffle : public cpu::OperatorOnCPU<base::PixelShuffle> {
        public:
            using self = PixelShuffle;
            using supper = cpu::OperatorOnCPU<base::PixelShuffle>;

            PixelShuffle();
            void init() override;
            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;
            void pixel_shuffle(const Tensor& x, Tensor& out, int scale_factor, std::string mode) override;

        private:
            std::string m_mode;
            int m_scale_factor = 1;

            pthreadpool_t m_threadpool = nullptr;
            xnn_status m_status = xnn_status_invalid_state;
            xnn_operator_t m_op = nullptr;
            std::shared_ptr<xnn_operator> m_shared_op;
        };
    }
}

#endif //TENNIS_KERNELS_XNNPACK_PIXELSHUFFLE_H
