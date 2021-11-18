//
// Created by sen on 2021/10/22.
//

#ifndef TENNIS_SUB_H
#define TENNIS_SUB_H

#include <kernels/cpu/operator_on_cpu.h>
#include "kernels/xnnpack/xnnpack.h"

namespace ts {
    namespace xnn {
        class Sub : public cpu::OperatorOnCPU<OperatorOnDevice> {
        public:
            using self = Sub;
            using supper = OperatorOnCPU;

            Sub() = default;

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            void sub(const Tensor &lhs, const Tensor &rhs, Tensor &out);

        private:
            pthreadpool_t m_threadpool;
            xnn_status m_status;
            xnn_operator_t m_op = nullptr;

            std::shared_ptr<xnn_operator> m_shared_op;
        };
    }
}

#endif //TENNIS_SUB_H
