//
// Created by sen on 2021/10/22.
//

#ifndef TENNIS_CONV2DV2_H
#define TENNIS_CONV2DV2_H

#include "kernels/xnnpack/xnnpack.h"
#include "kernels/cpu/operator_on_cpu.h"
#include <backend/common_structure.h>

namespace ts {
    namespace xnn {
        class Conv2DV2 : public cpu::OperatorOnCPU<OperatorOnDevice> {
        public:
            using self = Conv2DV2;
            using supper = OperatorOnCPU<OperatorOnDevice>;

            Conv2DV2();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            Operator::shared m_op_xnn_conv2d;
            Tensor m_int_padding4x2;    // save pre set padding
        };
    }
}

#endif //TENNIS_CONV2DV2_H
