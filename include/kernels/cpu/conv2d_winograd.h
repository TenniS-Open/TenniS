#ifndef TENSORSTACK_KERNELS_CPU_CONV2D_WINOGRAD_H
#define TENSORSTACK_KERNELS_CPU_CONV2D_WINOGRAD_H

#include "operator_on_cpu.h"
#include "backend/base/base_conv2d_winograd.h"

namespace ts {
    namespace cpu{
        class Conv2DWinograd : public OperatorOnCPU<base::Conv2DWinograd> {
        public:
            using self = Conv2DWinograd;
            using supper = base::Conv2DWinograd;

            //void conv2d_winograd(const Tensor &x, WinogradConv2DModel winograd_model, const Padding2D &padding, float padding_value,
            //    const Tensor &w, Conv2DFormat format, Tensor &out, Stack &stack);
            void conv2d_winograd(const Tensor &x, WinogradConv2DModel winograd_model,
                const Tensor &w, Conv2DFormat format, Tensor &out, Stack &stack);
        };
    }
}

#endif //TENSORSTACK_KERNELS_CPU_CONV2D_WINOGRAD_H