//
// Created by sen on 2021/10/22.
//

#ifndef TENNIS_ADD_H
#define TENNIS_ADD_H

#include "kernels/xnnpack/xnnpack.h"
#include "kernels/cpu/operator_on_cpu.h"

namespace ts {
    namespace xnn {
        class Add : public cpu::OperatorOnCPU<OperatorOnDevice> {
        public:
            using self = Add;
            using supper = OperatorOnCPU;

            Add() = default;

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            void add(const Tensor &lhs, const Tensor &rhs, Tensor &out);

        private:
            pthreadpool_t m_threadpool;
            xnn_status m_status;
            xnn_operator_t m_op = nullptr;
            std::shared_ptr<xnn_operator> m_shared_op;
        };
    }
}

#endif //TENNIS_ADD_H
