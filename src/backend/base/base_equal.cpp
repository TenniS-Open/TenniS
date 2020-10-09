#include <frontend/intime.h>
#include "backend/base/base_equal.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {

        bool uncorr_cast_dtype(DTYPE type) {
            if (type == 0 || type == 12 || (type > 15 && type != 21)) return true;
            else return false;
        }

        DTYPE upcast_dtype(Operator *op, Tensor &lhs, Tensor &rhs) {
            // diagonal element represents the type in dtype.h
            DTYPE lhs_dtype = lhs.dtype();
            DTYPE rhs_dtype = rhs.dtype();
            if (lhs_dtype == rhs_dtype) return lhs_dtype;
            else if (uncorr_cast_dtype(lhs_dtype) || uncorr_cast_dtype(rhs_dtype)) {
                TS_LOG_ERROR << "[" << op->op() << ":" << op->name() << "] Can not reduce mismatch type: "
                             << type_str(lhs.dtype()) << " vs. "
                             << type_str(rhs.dtype()) << eject;
            } else {
                static const char lookup_table[22][22]{
                        {0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0, 0, 0, 0, 0, 0},
                        {0, 1,  1,  3,  3,  5,  5,  7,  7,  9,  10, 11, 0, 1,  3,  5,  0, 0, 0, 0, 0, 1},
                        {0, 1,  2,  3,  3,  5,  6,  7,  7,  9,  10, 11, 0, 1,  3,  5,  0, 0, 0, 0, 0, 2},
                        {0, 3,  3,  3,  3,  5,  5,  7,  7,  9,  10, 11, 0, 3,  3,  5,  0, 0, 0, 0, 0, 3},
                        {0, 3,  3,  3,  4,  5,  6,  7,  8,  9,  10, 11, 0, 3,  3,  5,  0, 0, 0, 0, 0, 4},
                        {0, 5,  5,  5,  5,  5,  5,  7,  7,  10, 10, 11, 0, 5,  5,  5,  0, 0, 0, 0, 0, 5},
                        {0, 5,  6,  5,  6,  5,  6,  7,  8,  10, 10, 11, 0, 5,  5,  5,  0, 0, 0, 0, 0, 6},
                        {0, 7,  7,  7,  7,  7,  7,  7,  7,  11, 11, 11, 0, 7,  7,  7,  0, 0, 0, 0, 0, 7},
                        {0, 7,  7,  7,  8,  7,  8,  7,  8,  11, 11, 11, 0, 7,  7,  7,  0, 0, 0, 0, 0, 8},
                        {0, 9,  9,  9,  9,  10, 10, 11, 11, 9,  10, 11, 0, 9,  9,  10, 0, 0, 0, 0, 0, 9},
                        {0, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 11, 0, 10, 10, 11, 0, 0, 0, 0, 0, 10},
                        {0, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 0, 11, 11, 11, 0, 0, 0, 0, 0, 11},
                        {0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0, 0, 0, 0, 0, 0},
                        {0, 1,  1,  3,  3,  5,  5,  7,  7,  9,  10, 11, 0, 13, 14, 15, 0, 0, 0, 0, 0, 13},
                        {0, 3,  3,  3,  3,  5,  5,  7,  7,  9,  10, 11, 0, 14, 14, 15, 0, 0, 0, 0, 0, 14},
                        {0, 5,  5,  5,  5,  5,  5,  7,  7,  10, 11, 11, 0, 15, 15, 15, 0, 0, 0, 0, 0, 15},
                        {0, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 0, 13, 14, 15, 0, 0, 0, 0, 0, 21},
                        {0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0, 0, 0, 0, 0, 0},
                        {0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0, 0, 0, 0, 0, 0},
                        {0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0, 0, 0, 0, 0, 0},
                        {0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0, 0, 0, 0, 0, 0},
                        {0, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 0, 13, 14, 15, 0, 0, 0, 0, 0, 21}
                };
                return DTYPE(lookup_table[lhs_dtype][rhs_dtype]);
            }
        }

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
