#ifndef TENNIS_SOFTPLUS_H
#define TENNIS_SOFTPLUS_H

#include "backend/base/base_activation.h"
#include "kernels/cpu/operator_on_cpu.h"
#include "pthreadpool.h"

namespace ts {
    namespace xnn {
        class Softplus : public cpu::OperatorOnCPU<base::Activation> {
            public:
                using self = Softplus;
                using supper = cpu::OperatorOnCPU<base::Activation>;

                void init() override;
                void active(const Tensor& x, Tensor& out) override;

            private:
                pthreadpool_t m_pool = nullptr;
        };
    }
}

#endif //TENNIS_SOFTPLUS_H
