#ifndef TENSORSTACK_KERNELS_CPU_MOD_H
#define TENSORSTACK_KERNELS_CPU_MOD_H

#include <core/tensor.h>
#include "operator_on_cpu.h"
#include <backend/base/element_wise_reduce.h>

namespace ts {
    namespace cpu {
        class Mod : public OperatorOnCPU<ElementWiseReduce> {
        public:
            using self = Mod;
            using supper = OperatorOnCPU<ElementWiseReduce>;

            void reduce_with_broadcast(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_bias(const Tensor &lhs, const Tensor &rhs, Tensor &out, int dim) override;

            void reduce_with_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_bias_cross(const Tensor &lhs, const Tensor &rhs, Tensor &out, int dim) override;

            void reduce_with_scalar_cross(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_CPU_MOD_H
