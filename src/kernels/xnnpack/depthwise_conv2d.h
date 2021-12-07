//
// Created by sen on 2021/10/22.
//

#ifndef TENNIS_DEPTHWISECONV2D_H
#define TENNIS_DEPTHWISECONV2D_H

#include "kernels/xnnpack/xnnpack.h"
#include "kernels/cpu/operator_on_cpu.h"
#include <valarray>
#include <backend/common_structure.h>

namespace ts {
    namespace xnn {
        class DepthwiseConv2D : public cpu::OperatorOnCPU<OperatorOnDevice> {
        public:
            using self = DepthwiseConv2D;
            using supper = OperatorOnCPU;

            DepthwiseConv2D();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            void depthwise_conv2d(const Tensor &x, const Padding2D &padding, float padding_value,
                        const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                        Conv2DFormat format, Tensor &out);
        private:
            pthreadpool_t m_threadpool;
            xnn_status m_status;
            xnn_operator_t m_op = nullptr;

            std::shared_ptr<xnn_operator> m_shared_op;

            Tensor m_bias;
            Conv2DFormat m_format;
            std::valarray<int> m_padding4x2;
            float m_padding_value;
            std::valarray<int> m_stride4;
            std::valarray<int> m_dilation4;
            bool m_kernel_packed = false;
            float m_value_max = +std::numeric_limits<float>::infinity();
            float m_value_min = -std::numeric_limits<float>::infinity();
        };
    }
}

#endif //TENNIS_DEPTHWISECONV2D_H
