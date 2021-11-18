//
// Created by sen on 2021/10/22.
//

#ifndef TENNIS_RELU_H
#define TENNIS_RELU_H

#include "kernels/cpu/operator_on_cpu.h"
#include "kernels/xnnpack/xnnpack.h"


namespace ts {
    namespace xnn {
        class ReLU : public cpu::OperatorOnCPU<OperatorOnDevice> {
        public:
            using self = ReLU;
            using supper = OperatorOnCPU;

            ReLU() = default;

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            void relu(const Tensor &x, Tensor &out);

        private:
            pthreadpool_t m_threadpool;
            xnn_status m_status;
            xnn_operator_t m_op = nullptr;
        };
    }
}

#endif //TENNIS_RELU_H
