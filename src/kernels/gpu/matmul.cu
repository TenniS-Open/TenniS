//
// Created by sen on 2021/2/25.
//

#include <numeric>
#include "kernels/gpu/matmul.h"

namespace ts {
    namespace gpu {
        void MatMul::matmul_compute(Stack &stack, Tensor &a, Tensor &b, Tensor &out) {
            if (a.dims() == b.dims()) {
                stack.push(a);
                stack.push(b);
                RunOperator(m_op_inner_prod, stack, 2);
                out = stack.top()->clone();
                stack.erase(-1);
            } else {
                Shape output_shape = a.sizes();
                output_shape[a.dims() - 1] = b.size(1);
                Tensor reshaped_a = a.reshape({std::accumulate(a.sizes().begin(), a.sizes().end() - 1, 1, std::multiplies<int>()), a.size(a.dims() - 1)});
                stack.push(reshaped_a);
                stack.push(b);
                RunOperator(m_op_inner_prod, stack, 2);
                out = stack.top()->clone();
                out = out.reshape(output_shape);
                stack.erase(-1);
            }
        }
    }
}

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(MatMul, ts::GPU, "matmul")
