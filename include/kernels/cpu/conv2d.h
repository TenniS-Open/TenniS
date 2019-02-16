#ifndef TENSORSTACK_KERNELS_CPU_CONCAT_H
#define TENSORSTACK_KERNELS_CPU_CONCAT_H

#include "operator_on_cpu.h"
#include "backend/base/base_conv2d.h"


namespace ts {
	namespace cpu {
		class Conv2D : public OperatorOnAny<base::Conv2D> {
		public:
			using self = Conv2D;
			using supper = OperatorOnAny<base::Conv2D>;

            Conv2D() = default;

            void conv2d(const Tensor &x, const Padding2D &padding, float padding_value,
                        const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                        Conv2DFormat format, Tensor &out, Stack &stack) override;
		};
	}
}


#endif //TS_KERNELS_CONCAT_H