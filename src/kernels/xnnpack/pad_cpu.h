#ifndef TENNIS_KERNELS_XNNPACK_PAD_H
#define TENNIS_KERNELS_XNNPACK_PAD_H

#include "kernels/cpu/operator_on_cpu.h"
#include "backend/base/base_pad.h"


namespace ts {
    namespace xnn {
        class PadOnCPU : public ts::cpu::OperatorOnCPU<base::Pad> {
        public:
            using self = PadOnCPU;
            using supper = ts::cpu::OperatorOnCPU<base::Pad>;

            void pad(const Tensor &x, const std::vector<std::array<int, 2>> &padding, float padding_value,
                     Tensor &out) override;
        };
    }
}

#endif  // TENNIS_KERNELS_XNNPACK_PAD_H
