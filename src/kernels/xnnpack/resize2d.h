#ifndef TENNIS_KERNELS_XNNPACK_RESIZE2D_H
#define TENNIS_KERNELS_XNNPACK_RESIZE2D_H

#include "backend/base/base_resize2d.h"
#include "kernels/cpu/operator_on_cpu.h"
#include "pthreadpool.h"

namespace ts {
    namespace xnn {
        class Resize2D : public ts::cpu::OperatorOnCPU<base::Resize2D> {
        public:
            using self = Resize2D;
            using supper = ts::cpu::OperatorOnCPU<base::Resize2D>;

            void init() override;
            void resize2d(const Tensor &x, int dim, Resize2DType type, Tensor &out) override;

        private:
            pthreadpool_t m_pool;
        };
    }
}

#endif  // TENNIS_KERNELS_XNNPACK_RESIZE2D_H
