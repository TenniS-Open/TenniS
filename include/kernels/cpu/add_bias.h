#ifndef TS_KERNELS_ADD_BIAS_H
#define TS_KERNELS_ADD_BIAS_H

#include <runtime/operator.h>
#include <core/tensor.h>
#include <runtime/stack.h>

#include <backend/base/base_add_bias.h>
#include "operator_on_cpu.h"

namespace ts {
    namespace cpu {
        class AddBias : public OperatorOnCPU<base::AddBias> {
        public:
            using self = AddBias;
            using supper = OperatorOnCPU<base::AddBias>;

            AddBias() = default;

            void add(const Tensor &x, const Tensor &b, int dim, Tensor &out) override;
        };
    }
}

#endif
