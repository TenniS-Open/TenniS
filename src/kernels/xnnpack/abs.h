//
// Created by sen on 2021/10/22.
//

#ifndef TENNIS_ABS_H
#define TENNIS_ABS_H

#include "kernels/xnnpack/xnnpack.h"
#include "kernels/cpu/operator_on_cpu.h"

namespace ts {
    namespace xnn {
        class Abs : public cpu::OperatorOnCPU<OperatorOnDevice> {
        public:
            using self = Abs;
            using supper = OperatorOnDevice;

            Abs() = default;

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            void abs(const Tensor &x, Tensor &out);
        private:
            xnn_status m_status;
            xnn_operator_t m_op = nullptr;
            pthreadpool_t  m_threadpool = nullptr;
            std::shared_ptr<xnn_operator> m_shared_op;
        };
    }
}

#endif //TENNIS_ABS_H
