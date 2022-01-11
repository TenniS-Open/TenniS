//
// Created by sen on 2022/1/6.
//

#ifndef TENNIS_TRANSPOSE_H
#define TENNIS_TRANSPOSE_H

#include "kernels/cpu/transpose.h"
#include "kernels/xnnpack/pthreadpool.h"

namespace ts {
    namespace xnn {
    class Transpose : public cpu::OperatorOnCPU<base::Transpose> {
        public:
            using self = Transpose;
            using supper = ts::base::Transpose;

            void init() override;

            void transpose(const Tensor &x, const std::vector<int> &permute, Tensor &out) override;

        private:
            pthreadpool_t m_pool;
        };
    }
}

#endif //TENNIS_TRANSPOSE_H
