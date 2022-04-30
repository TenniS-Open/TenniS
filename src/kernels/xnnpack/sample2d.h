#ifndef TENNIS_XNN_SAMPLE2D_H
#define TENNIS_XNN_SAMPLE2D_H

#include "kernels/cpu/operator_on_cpu.h"
#include "kernels/xnnpack/xnnpack.h"


namespace ts {
    namespace xnn {
        class Sample2D : public cpu::OperatorOnCPU<OperatorOnDevice> {
        public:
            using self = Sample2D;
            using supper = OperatorOnCPU;

            Sample2D();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            void sample2d(const Tensor &x, Tensor &out);

        private:
            int m_dim;
            float m_scale;
            Operator::shared m_sample_op;

            Tensor m_sample_size;
            Tensor m_sample_affine;
        };
    }
}

#endif //TENNIS_XNN_SAMPLE2D_H
