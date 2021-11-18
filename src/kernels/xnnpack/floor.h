//
// Created by sen on 2021/10/22.
//

#ifndef TENNIS_FLOOR_H
#define TENNIS_FLOOR_H

#include "kernels/xnnpack/xnnpack.h"
#include "kernels/cpu/operator_on_cpu.h"

namespace ts {
    namespace xnn {
        class Floor : public cpu::OperatorOnCPU<OperatorOnDevice> {
        public:
            using self = Floor;
            using supper = OperatorOnDevice;

            Floor() = default;

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            void floor(const Tensor &x, Tensor &out);
        private:
            xnn_status m_status;
            xnn_operator_t m_op = nullptr;
            pthreadpool_t  m_threadpool = nullptr;
        };
    }
}

#endif //TENNIS_FLOOR_H
