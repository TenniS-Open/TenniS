//
// Created by sen on 2021/10/22.
//

#ifndef TENNIS_RELUMAX_H
#define TENNIS_RELUMAX_H

#include "kernels/cpu/operator_on_cpu.h"
#include "kernels/xnnpack/xnnpack.h"


namespace ts {
    namespace xnn {
        class ReLUMax : public cpu::OperatorOnCPU<OperatorOnDevice> {
        public:
            using self = ReLUMax;
            using supper = OperatorOnCPU;

            ReLUMax();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            void relu_max(const Tensor &x, Tensor &out);

        private:
            pthreadpool_t m_threadpool;
            xnn_status m_status;
            xnn_operator_t m_op = nullptr;

            float m_max;
        };
    }
}

#endif //TENNIS_RELUMAX_H
