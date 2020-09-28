#include "backend/base/base_equal.h"
#include "core/tensor_builder.h"
#define UPCAST_DTYPE

namespace ts {
    namespace base {

#ifdef UPCAST_DTYPE
        DTYPE upcast_dtype(Tensor &lhs, Tensor &rhs) {
            // diagonal element represents the type in dtype.h
            int lookup_table[15][15]{
                    {1	,1	,3	,3	,5	,5	,7	,7	,9	,10	,11	,1	,3	,5	,1},
                    {1	,2	,3	,3	,5	,6	,7	,7	,9	,10	,11	,1	,3	,5	,2},
                    {3	,3	,3	,3	,5	,5	,7	,7	,9	,10	,11	,3	,3	,5	,3},
                    {3	,3	,3	,4	,5	,6	,7	,8	,9	,10	,11	,3	,3	,5	,4},
                    {5	,5	,5	,5	,5	,5	,7	,7	,10	,10	,11	,5	,5	,5	,5},
                    {5	,6	,5	,6	,5	,6	,7	,8	,10	,10	,11	,5	,5	,5	,6},
                    {7	,7	,7	,7	,7	,7	,7	,7	,11	,11	,11	,7	,7	,7	,7},
                    {7	,7	,7	,8	,7	,8	,7	,8	,11	,11	,11	,7	,7	,7	,8},
                    {9	,9	,9	,9	,10	,10	,11	,11	,9	,10	,11	,9	,9	,10	,9},
                    {10	,10	,10	,10	,10	,10	,11	,11	,10	,10	,11	,10	,10	,11	,10},
                    {11	,11	,11	,11	,11	,11	,11	,11	,11	,11	,11	,11	,11	,11	,11},
                    {1	,1	,3	,3	,5	,5	,7	,7	,9	,10	,11	,13	,14	,15	,13},
                    {3	,3	,3	,3	,5	,5	,7	,7	,9	,10	,11	,14	,14	,15	,14},
                    {5	,5	,5	,5	,5	,5	,7	,7	,10	,11	,11	,15	,15	,15	,15},
                    {1	,2	,3	,4	,5	,6	,7	,8	,9	,10	,11	,13	,14	,15	,21}
            };
            if (lhs.dtype() == rhs.dtype()) return lhs.dtype();
            return DTYPE(lookup_table[lhs.dtype()][rhs.dtype()]);
        }
#endif

        int Equal::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto &lhs = *stack.index(0);
            auto &rhs = *stack.index(1);

#ifdef UPCAST_DTYPE
            DTYPE out_dtype = upcast_dtype(lhs, rhs);
            if (lhs.dtype() != out_dtype) {
                lhs = tensor::cast(out_dtype, lhs);
            }
            if (rhs.dtype() != out_dtype) {
                rhs = tensor::cast(out_dtype, rhs);
            }
#endif

            if (lhs.dtype() != rhs.dtype()) {
                TS_LOG_ERROR << "[" << this->op() << ":" << this->name() << "] Can not reduce mismatch type: "
                             << type_str(lhs.dtype()) << " vs. "
                             << type_str(rhs.dtype()) << eject;
            }

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

            auto &lhs = stack[0];
            auto &rhs = stack[1];

            auto lhs_shape = lhs.sizes();
            auto rhs_shape = rhs.sizes();
            Shape out_shape;

            bool do_broadcast = reduce(this, lhs_shape, rhs_shape, out_shape, true);

            auto memory_device = running_memory_device();

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
