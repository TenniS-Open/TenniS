#ifndef TENNIS_SOFTPLUS_H
#define TENNIS_SOFTPLUS_H

#include "backend/base/base_activation.h"
#include "operator_on_cpu.h"

namespace ts {
    namespace cpu {
        class Softplus : public OperatorOnCPU<base::Activation> {
        public:
            using self = Softplus;
            using supper = OperatorOnCPU<base::Activation>;

            void active(const Tensor& x, Tensor& out) override;
        };
    }
}

#endif //TENNIS_SOFTPLUS_H
