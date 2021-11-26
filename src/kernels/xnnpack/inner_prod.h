//
// Created by sen on 2021/11/17.
//

#ifndef TENNIS_INNER_PROD_H
#define TENNIS_INNER_PROD_H

#include "kernels/xnnpack/xnnpack.h"
#include "kernels/cpu/operator_on_cpu.h"

namespace ts {
    namespace xnn {
        class InnerProd : public cpu::OperatorOnCPU<OperatorOnDevice> {
        public:
            using self = InnerProd;
            using supper = OperatorOnDevice;

            InnerProd();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            void inner_prod(const Tensor &A, const Tensor &B, const Tensor &C, Tensor &out);
        private:
            xnn_status m_status;
            xnn_operator_t m_op = nullptr;
            pthreadpool_t  m_threadpool = nullptr;

            std::unordered_map<size_t, std::shared_ptr<xnn_operator>> m_shared_op_map;
        };
    }
}
#endif //TENNIS_INNER_PROD_H
