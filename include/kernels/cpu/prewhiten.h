#ifndef TENSORSTACK_KERNELS_CPU_PREWHITEN_H
#define TENSORSTACK_KERNELS_CPU_PREWHITEN_H

#include "backend/base/base_prewhiten.h"
#include "operator_on_cpu.h"

namespace ts {
	namespace cpu {
		class PreWhiten : public OperatorOnCPU<base::PreWhiten> {
		public:
			using supper = ts::Operator;

			void prewhiten(const Tensor &x, Tensor &out) override;
		};
	}
}



#endif //TENSORSTACK_KERNELS_CPU_PREWHITEN_H