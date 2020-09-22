#include "backend/base/base_equal.h"

namespace ts {
    namespace base {
        int Equal::run(Stack &stack) {
            std::vector<Tensor::Prototype> output;

            infer(stack, output);

            auto &lhs = stack[0];
            auto &rhs = stack[1];

            auto lhs_shape = lhs.sizes();
            auto rhs_shape = rhs.sizes();
            Shape out_shape;

            bool do_broadcast = reduce(this, lhs_shape, rhs_shape, out_shape, true);

            auto memory_device = running_memory_device();

            output[0] = Tensor::Prototype(BOOLEAN, out_shape);
            auto &out = *stack.push(output[0], memory_device);

            if (!do_broadcast) {
                reduce_with_same_shape(lhs, rhs, out);
            } else if (is_scalar(rhs_shape)) {
                reduce_with_scalar(lhs, rhs, out);
            } else if (is_scalar(lhs_shape)) {
                reduce_with_scalar_cross(lhs, rhs, out);
            } else {
                reduce_with_broadcast(lhs, rhs, out);
            }

            return 1;
        }
    }
}
