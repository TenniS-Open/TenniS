#ifndef TENNIS_KERNELS_GPU_EQUAL_H
#define TENNIS_KERNELS_GPU_EQUAL_H

#include "backend/base/base_equal.h"
#include "operator_on_gpu.h"

namespace ts {
    namespace gpu {
        class Equal : public OperatorOnGPU<base::Equal> {
            using self = Equal;
            using supper = OperatorOnGPU<base::Equal>;

            void reduce_with_broadcast(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_scalar_cross(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;
        };
    }
}

#endif //TENNIS_KERNELS_GPU_EQUAL_H
