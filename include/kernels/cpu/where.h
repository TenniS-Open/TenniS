#ifndef TENNIS_WHERE_H
#define TENNIS_WHERE_H

#include "operator_on_cpu.h"
#include "backend/base/base_where.h"

namespace ts {
    namespace cpu {
        class Where : public OperatorOnCPU<base::Where> {
        public:
            using self = Where;
            using supper = OperatorOnCPU<base::Where>;

            void reduce_with_broadcast(const Tensor &cond, const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_same_shape(const Tensor &cond, const Tensor &lhs, const Tensor &rhs, Tensor &out) override;
        };
    }
}

#endif //TENNIS_WHERE_H
