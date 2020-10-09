#ifndef TENNIS_SOFTPLUS_H
#define TENNIS_SOFTPLUS_H

#include "backend/base/base_activation.h"
#include "operator_on_gpu.h"

namespace ts {
    namespace gpu {
        class Softplus : public OperatorOnGPU<base::Activation> {
        public:
            using self = Softplus;
            using supper = OperatorOnGPU<base::Activation>;

            void active(const Tensor &x, Tensor &out) override;
        };
    }
}

#endif //TENNIS_SOFTPLUS_H
