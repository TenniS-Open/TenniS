#ifndef TENNIS_KERNELS_GPU_SCATTER_ND_H
#define TENNIS_KERNELS_GPU_SCATTER_ND_H

#include <core/tensor.h>
#include <runtime/stack.h>
#include "operator_on_gpu.h"
#include <backend/base/base_scatter_nd.h>

namespace ts {
    namespace gpu {
        class ScatterND : public OperatorOnGPU<base::ScatterND> {
        public:
            using self = ScatterND;
            using supper = OperatorOnGPU<base::ScatterND>;

            void scatter(const Tensor &data, const Tensor &indices, Tensor &updates, Tensor &out) override;
        };
    }
}

#endif  // TENNIS_KERNELS_GPU_SCATTER_ND_H
