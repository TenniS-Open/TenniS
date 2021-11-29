//
// Created by sen on 2021/10/22.
//

#ifndef TENNIS_SQUARE_H
#define TENNIS_SQUARE_H

#include "kernels/xnnpack/xnnpack.h"
#include "kernels/cpu/operator_on_cpu.h"

namespace ts {
    namespace xnn {
        class Square : public cpu::OperatorOnCPU<OperatorOnDevice> {
        public:
            using self = Square;
            using supper = OperatorOnDevice;

            Square() = default;

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            void square(const Tensor &x, Tensor &out);
        private:
            xnn_status m_status;
            xnn_operator_t m_op = nullptr;
            pthreadpool_t m_threadpool = nullptr;
            std::shared_ptr<xnn_operator> m_shared_op;
        };
    }
}

#endif //TENNIS_SQUARE_H
