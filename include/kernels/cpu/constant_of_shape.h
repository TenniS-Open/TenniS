#ifndef TENNIS_CONSTANT_OF_SHAPE_H
#define TENNIS_CONSTANT_OF_SHAPE_H

#include "operator_on_cpu.h"

namespace ts {
    namespace cpu {
        class ConstantOfShape : public OperatorOnAny<OperatorOnDevice> {
            using self = ConstantOfShape;
            using supper = OperatorOnAny<OperatorOnDevice>;
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
