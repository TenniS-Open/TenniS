#ifndef TENNIS_KERNELS_XNNPACK_AFFINE_SAMPLE2D_H
#define TENNIS_KERNELS_XNNPACK_AFFINE_SAMPLE2D_H

#include "kernels/cpu/operator_on_cpu.h"
#include "backend/base/base_affine_sample2d.h"
#include "pthreadpool.h"


namespace ts {
    namespace xnn {
        class Affine_Sample2D : public ts::cpu::OperatorOnCPU<base::Affine_Sample2D> {
        public:
            using self = Affine_Sample2D;
            using supper = ts::cpu::OperatorOnCPU<base::Affine_Sample2D>;

            Affine_Sample2D() = default;

            void init() override;

            void affine_sample_run(const Tensor &x, float rz00, float rz01, float rz02, float rz10, float rz11, float rz12,
                                           float rz20, float rz21, float rz22, Affine_Sample2DType type, int dim,
                                           base::AffineOuterMode outer_mode, float outer_value,
                                           Tensor &out) override;

        private:
            pthreadpool_t m_pool = nullptr;
        };
    }
}

#endif //TENNIS_KERNELS_XNNPACK_AFFINE_SAMPLE2D_H
