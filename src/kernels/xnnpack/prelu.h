//
// Created by sen on 2021/10/22.
//

#ifndef TENNIS_PRELU_H
#define TENNIS_PRELU_H

#include "kernels/xnnpack/xnnpack.h"
#include "kernels/cpu/operator_on_cpu.h"

namespace ts {
    namespace xnn {
        class PReLU : public cpu::OperatorOnCPU<OperatorOnDevice> {
        public:
            using self = PReLU;
            using supper = OperatorOnDevice;

            PReLU();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            void prelu(const Tensor &x, const Tensor &slope, Tensor &out);

        private:
            xnn_status m_status;
            xnn_operator_t m_op = nullptr;
            pthreadpool_t  m_threadpool;
            std::shared_ptr<xnn_operator> m_shared_op;
            int m_dim = -1;
            bool check_inputs(Stack &stack) const;
        };
    }
}

#endif //TENNIS_PRELU_H
