#ifndef TENNIS_KERNELS_XNNPACK_POOLING2D_V2_H
#define TENNIS_KERNELS_XNNPACK_POOLING2D_V2_H

#include "kernels/cpu/operator_on_cpu.h"


namespace ts {
	namespace xnn {
		class Pooling2DV2 : public cpu::OperatorOnCPU<Operator> {
		public:
			using self = Pooling2DV2;
			using supper = cpu::OperatorOnCPU<Operator>; 

			Pooling2DV2();

			void init() override;

			/**
			 * Stack has: x, padding, ksize, stride
			 */
			int run(Stack &stack) override;

			int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

		private:
			Operator::shared m_op_pooling2d;

			Tensor m_padding_int4x2;    // save pre set padding
			Tensor m_ksize_int4;
			Tensor m_stride_int4;
		};
	}
}


#endif //TENNIS_KERNELS_XNNPACK_POOLING2D_V2_H