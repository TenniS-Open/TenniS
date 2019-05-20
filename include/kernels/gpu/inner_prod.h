#ifndef TENSORSTACK_KERNELS_GPU_INNER_PROD_H
#define TENSORSTACK_KERNELS_GPU_INNER_PROD_H

#include "operator_on_gpu.h"
#include "backend/base/base_inner_prod.h"


namespace ts {
    namespace gpu {
        class InnerProd : public OperatorOnGPU<base::InnerProd> {
        public:
            using self = InnerProd;
            using supper = OperatorOnGPU<base::InnerProd>;

            void inner_prod(const Tensor &lhs, const Tensor &rhs, bool transpose, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_GPU_INNER_PROD_H
