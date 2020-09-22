#ifndef TENNIS_EQUAL_H
#define TENNIS_EQUAL_H

#include "operator_on_cpu.h"
#include "backend/base/base_equal.h"

namespace ts {
    namespace cpu {
        class Equal : public OperatorOnCPU<base::Equal> {
        public:
            using self = Equal;
            using supper = OperatorOnCPU<base::Equal>;

            void reduce_with_broadcast(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_bias(const Tensor &lhs, const Tensor &rhs, Tensor &out) = delete;

            void reduce_with_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_bias_cross(const Tensor &lhs, const Tensor &rhs, Tensor &out) = delete;

            void reduce_with_scalar_cross(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

        };
    }
}


#endif //TENNIS_EQUAL_H
