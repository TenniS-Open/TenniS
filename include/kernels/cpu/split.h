#ifndef TENNIS_KERNELS_CPU_CONCAT_H
#define TENNIS_KERNELS_CPU_CONCAT_H

#include "operator_on_cpu.h"
#include "backend/base/base_split.h"


namespace ts {
	namespace cpu {
		class SplitOp : public OperatorOnAny<base::Split> {
		public:
			using self = SplitOp;
			using supper = OperatorOnAny<base::Split>;

            SplitOp() = default;

            void split(const Tensor &x, const std::vector<int> &split, int dim,
                       std::vector<Tensor> &out) override;
		};
	}
}


#endif //TENNIS_KERNELS_CPU_CONCAT_H