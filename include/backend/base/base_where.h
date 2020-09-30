#ifndef TENNIS_BASE_WHERE_H
#define TENNIS_BASE_WHERE_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class Where : public OperatorOnDevice {
        public:
            using self = Where;
            using supper = OperatorOnDevice;

            Where() = default;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            int run(Stack &stack) override;

            virtual void
            reduce_with_broadcast(const Tensor &cond, const Tensor &lhs, const Tensor &rhs, Tensor &out) = 0;

            virtual void
            reduce_with_same_shape(const Tensor &cond, const Tensor &lhs, const Tensor &rhs, Tensor &out) = 0;

            virtual bool
            reduce(Operator *op, Shape &_cond_shape, Shape &_lhs_shape, Shape &_rhs_shape, Shape &_out_shape,
                   bool broadcast);
        };
    }
}

#endif //TENNIS_BASE_WHERE_H
