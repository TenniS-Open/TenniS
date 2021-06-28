#ifndef TENNIS_KERNELS_CPU_MATMUL_H
#define TENNIS_KERNELS_CPU_MATMUL_H

#include <core/tensor.h>
#include <runtime/stack.h>
#include "operator_on_cpu.h"
#include <backend/base/base_matmul.h>

namespace ts {
    namespace cpu {
        class MatMul : public OperatorOnCPU<base::MatMul> {
        public:
            using self = MatMul;
            using supper = OperatorOnCPU<base::MatMul>;

            void matmul_compute(Stack &stack, Tensor &a, Tensor &b, Tensor &out) override;
        };
    }
}

#endif  // TENNIS_KERNELS_CPU_MATMUL_H
