//
// Created by sen on 2021/10/22.
//

#ifndef TENNIS_LEAKY_RELU_H
#define TENNIS_LEAKY_RELU_H

#include "kernels/xnnpack/xnnpack.h"
#include "kernels/cpu/operator_on_cpu.h"

namespace ts {
    namespace xnn {
        class LeakyReLU : public cpu::OperatorOnCPU<OperatorOnDevice> {
        public:
            using self = LeakyReLU;
            using supper = OperatorOnDevice;

            LeakyReLU();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            void leaky_relu(const Tensor &x, Tensor &out);

        private:
            xnn_status m_status;
            xnn_operator_t m_op = nullptr;
            pthreadpool_t m_threadpool;
            std::shared_ptr<xnn_operator> m_shared_op;
            float m_scale;
        };
    }
}

#endif //TENNIS_LEAKY_RELU_H
