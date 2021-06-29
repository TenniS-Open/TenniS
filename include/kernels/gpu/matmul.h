#ifndef TENNIS_KERNELS_GPU_MATMUL_H
#define TENNIS_KERNELS_GPU_MATMUL_H

#include <core/tensor.h>
#include <runtime/stack.h>
#include "operator_on_gpu.h"
#include <backend/base/base_matmul.h>

namespace ts {
    namespace gpu {
        class MatMul : public OperatorOnGPU<base::MatMul> {
        public:
            using self = MatMul;
            using supper = OperatorOnGPU<base::MatMul>;

            void matmul_compute(Stack &stack, Tensor &a, Tensor &b, Tensor &out) override;
        };
    }
}

#endif  // TENNIS_KERNELS_GPU_MATMUL_H
