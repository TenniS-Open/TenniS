#include <frontend/intime.h>
#include "backend/base/base_equal.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        int Equal::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto lhs = *stack.index(0);
            auto rhs = *stack.index(1);

            auto lhs_shape = lhs.sizes();
            auto rhs_shape = rhs.sizes();
            Shape out_shape;

            bool do_broadcast = reduce(this, lhs_shape, rhs_shape, out_shape, true);
            (void)(do_broadcast);

            output.resize(1);
            output[0] = Tensor::Prototype(BOOLEAN, out_shape);

            return 1;
        }

        int Equal::run(Stack &stack) {
            std::vector<Tensor::Prototype> output;

            infer(stack, output);

            auto lhs = stack[0];
            auto rhs = stack[1];

            DTYPE out_dtype = upcast_dtype(this, lhs, rhs);
            lhs = intime::cast(lhs, out_dtype);
            rhs = intime::cast(rhs, out_dtype);

            auto lhs_shape = lhs.sizes();
            auto rhs_shape = rhs.sizes();
            Shape out_shape;

            bool do_broadcast = reduce(this, lhs_shape, rhs_shape, out_shape, true);

            auto memory_device = running_memory_device();

            lhs = lhs.view(memory_device);
            rhs = rhs.view(memory_device);
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
