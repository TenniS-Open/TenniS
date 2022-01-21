#ifndef TENNIS_XNN_EXP_H
#define TENNIS_XNN_EXP_H

#include "backend/base/base_activation.h"
#include "kernels/cpu/operator_on_cpu.h"
#include "pthreadpool.h"

namespace ts {
	namespace xnn {
        class Exp : public ts::cpu::OperatorOnCPU<base::Activation> {
		public:
		    using self = Exp;
			using supper = ts::cpu::OperatorOnCPU<base::Activation>;

            void init() override;
            void active(const Tensor &x, Tensor &out) override;

        private:
            pthreadpool_t m_pool = nullptr;
		};
	}
}



#endif //TENNIS_XNN_EXP_H
