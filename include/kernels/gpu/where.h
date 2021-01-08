#ifndef TENNIS_KERNELS_GPU_WHERE_H
#define TENNIS_KERNELS_GPU_WHERE_H

#include "operator_on_gpu.h"
#include "backend/base/base_where.h"

namespace ts {
    namespace gpu {
        class Where : public OperatorOnGPU<base::Where> {
        public:
            using self = Where;
            using supper = OperatorOnGPU<base::Where>;

            void reduce_with_broadcast(const Tensor &cond, const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_same_shape(const Tensor &cond, const Tensor &lhs, const Tensor &rhs, Tensor &out) override;
        };
    }
}

#endif //TENNIS_KERNELS_GPU_WHERE_H
