#ifndef TENNIS_KERNELS_CPU_LSTM_H
#define TENNIS_KERNELS_CPU_LSTM_H

#include "operator_on_cpu.h"
#include "backend/base/base_lstm.h"

namespace ts {
    namespace cpu {
        class LSTM : public OperatorOnCPU<base::LSTM> {
        public:
            using self = LSTM;
            using supper = OperatorOnCPU<base::LSTM>;

            void lstm_compute(Stack &stack, Tensor &x, Tensor &w, Tensor &r, Tensor &b,
                          Tensor &initial_h, Tensor &initial_c, int reverse, std::vector<Tensor> &out) override;
        };
    }
}

#endif //TENNIS_KERNELS_CPU_LSTM_H
