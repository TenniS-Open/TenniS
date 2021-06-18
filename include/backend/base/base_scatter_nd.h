#ifndef TENNIS_BASE_SCATTER_ND_H
#define TENNIS_BASE_SCATTER_ND_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class ScatterND : public OperatorOnDevice {
        public:
            using self = ScatterND;
            using supper = OperatorOnDevice;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void scatter(const Tensor &data, const Tensor &indices, Tensor &updates, Tensor &out) = 0;
        };
    }
}

#endif //TENNIS_BASE_SCATTER_ND_H
