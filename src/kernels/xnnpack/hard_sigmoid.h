//
// Created by sen on 2021/10/22.
//

#ifndef TENNIS_HARD_SIGMOID_H
#define TENNIS_HARD_SIGMOID_H

#include "kernels/xnnpack/xnnpack.h"
#include "kernels/cpu/operator_on_cpu.h"

namespace ts {
    namespace xnn {
        class HardSigmoid : public cpu::OperatorOnCPU<OperatorOnDevice> {
        public:
            using self = HardSigmoid;
            using supper = OperatorOnDevice;

            HardSigmoid();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            void hard_sigmoid(const Tensor &x, Tensor &out);
        private:
            xnn_status m_status;
            xnn_operator_t m_mul_op = nullptr;
            xnn_operator_t m_add_op = nullptr;
            pthreadpool_t m_threadpool = nullptr;
            std::shared_ptr<xnn_operator> m_shared_mul_op;
            std::shared_ptr<xnn_operator> m_shared_add_op;

            float m_alpha;
            float m_beta;
        };
    }
}

#endif //TENNIS_HARD_SIGMOID_H
