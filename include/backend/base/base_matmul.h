#ifndef TENNIS_BASE_MATMUL_H
#define TENNIS_BASE_MATMUL_H

#include "operator_on_device.h"
#include "utils/ctxmgr_lite.h"
#include "core/device_context.h"
#include "global/operator_factory.h"
#include "backend/name.h"

namespace ts {
    namespace base {
        class MatMul : public OperatorOnDevice {
        public:
            using self = MatMul;
            using supper = OperatorOnDevice;

            MatMul() = default;

            void init() override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            int run(Stack &stack) override;

            virtual void matmul_compute(Stack &stack, Tensor &a, Tensor &b, Tensor &out) = 0;

        protected:
            Operator::shared m_op_inner_prod;
        };
    }
}

#endif //TENNIS_BASE_MATMUL_H
