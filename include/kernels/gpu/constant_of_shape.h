#ifndef TENNIS_CONSTANT_OF_SHAPE_H
#define TENNIS_CONSTANT_OF_SHAPE_H

#include "kernels/cpu/constant_of_shape.h"
#include "operator_on_gpu.h"

namespace ts {
    namespace gpu {
        class ConstantOfShape : public OperatorOnGPU<Operator> {
            using self = ConstantOfShape;
            using supper = OperatorOnGPU<Operator>;
        public:
        public:
            ConstantOfShape();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void constant_of_shape(const Tensor &val, Tensor &out);

        private:
            Tensor m_tensor;
        };
    }
}

#endif //TENNIS_CONSTANT_OF_SHAPE_H
