//
// Created by sen on 2022/1/21.
//

#ifndef TENNIS_ERF_H
#define TENNIS_ERF_H

#include "kernels/xnnpack/pthreadpool.h"
#include "backend/base/base_activation.h"
#include "kernels/cpu/operator_on_cpu.h"

namespace ts {
    namespace xnn {
        class Erf : public cpu::OperatorOnCPU<base::Activation> {
        public:
            using self = Erf;
            using supper = ts::base::Activation;

            void init() override;
            void active(const Tensor &x, Tensor &out) final;
        private:
            pthreadpool_t m_pool = nullptr;
        };
    }
}

#endif //TENNIS_ERF_H
