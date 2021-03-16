#ifndef TENNIS_BACKEND_BASE_BASE_LSTM_H
#define TENNIS_BACKEND_BASE_BASE_LSTM_H

#include "operator_on_device.h"
#include "utils/ctxmgr_lite.h"
#include "core/device_context.h"
#include "global/operator_factory.h"
#include "backend/name.h"

namespace ts {
    namespace base {
        class LSTM : public OperatorOnDevice {
        public:
            using self = LSTM;
            using supper = OperatorOnDevice;

            LSTM();

            void init() override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            int run(Stack &stack) override;  // reference to gemm, set to device uncorrelated

            virtual void lstm_compute(Stack &stack, Tensor &x, Tensor &w, Tensor &r, Tensor &b,
                          Tensor &initial_h, Tensor &initial_c, int reverse, std::vector<Tensor> &out) = 0;

        protected:
            int m_direction = 0;  // 0:forward, 1:reverse, 2:bidirectional
            int m_hidden_size;
            Operator::shared m_op_gemm_notran;
            Operator::shared m_op_gemm_transb;
            Operator::shared m_op_mul;
            Operator::shared m_op_add;
            Operator::shared m_op_sigmoid;
            Operator::shared m_op_tanh;
            Operator::shared m_op_concat_dim0;
            Operator::shared m_op_concat_dim1;
        };
    }
}

#endif //TENNIS_BACKEND_BASE_BASE_LSTM_H
