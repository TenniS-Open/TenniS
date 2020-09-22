//
// Created by kier on 2019/1/28.
//

#include <backend/base/element_wise_reduce.h>

#include "backend/base/element_wise_reduce.h"

#include "utils/assert.h"
#include "runtime/stack.h"

#include <numeric>
#include "core/tensor_builder.h"
#define UPCAST_DTYPE

namespace ts {
    static inline void front_append_ones(Shape &shape, int count) {
        Shape ones(count, 1);
        shape.insert(shape.begin(), ones.begin(), ones.end());
    }

    bool ElementWiseReduce::reduce(Operator *op, Shape &_lhs_shape, Shape &_rhs_shape, Shape &_out_shape, bool broadcast) {

        if (_lhs_shape == _rhs_shape) {
            _out_shape = _lhs_shape;
            return false;
        }

        if (!broadcast) {
            TS_LOG_ERROR << "[" << op->op() << ":" << op->name() << "] Can not reduce shape: " << to_string(_lhs_shape) << " vs. " << to_string(_rhs_shape) << eject;
        }

        // TS_AUTO_CHECK(!_lhs_shape.empty());
        // TS_AUTO_CHECK(!_rhs_shape.empty());

        auto lhs_shape = _lhs_shape;
        auto rhs_shape = _rhs_shape;

        if (lhs_shape.size() > rhs_shape.size()) {
            front_append_ones(rhs_shape, int(lhs_shape.size() - rhs_shape.size()));
        } else if (lhs_shape.size() < rhs_shape.size()) {
            front_append_ones(lhs_shape, int(rhs_shape.size() - lhs_shape.size()));
        } else {
        }

        auto dims = lhs_shape.size();

        Shape out_shape(dims);
        bool do_broadcast = false;

        for (size_t i = 0; i < dims; ++i) {
            int size = lhs_shape[i];
            if (lhs_shape[i] != rhs_shape[i]) {
                if (lhs_shape[i] != 1 && rhs_shape[i] != 1)
					TS_LOG_ERROR << "[" << op->op() << ":" << op->name() << "] Can not reduce shape: " << to_string(_lhs_shape) << " vs. " << to_string(_rhs_shape) << eject;
                do_broadcast = true;
                size = lhs_shape[i] * rhs_shape[i];
            }
            out_shape[i] = size;
        }

        _lhs_shape = lhs_shape;
        _rhs_shape = rhs_shape;
        _out_shape = out_shape;

        return do_broadcast;
    }

    void ElementWiseReduce::init() {
        supper::init();
    }

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

    int ElementWiseReduce::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
        TS_AUTO_CHECK(stack.size() == 2);

        auto lhs = *stack.index(0);
        auto rhs = *stack.index(1);

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
        output[0] = Tensor::Prototype(lhs.dtype(), out_shape);

        return 1;
    }

    bool ElementWiseReduce::is_scalar(const Shape &shape) {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) == 1;
    }

    void ElementWiseReduce::reduce_with_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
        this->reduce_with_broadcast(lhs, rhs, out);
    }

    void ElementWiseReduce::reduce_with_bias(const Tensor &lhs, const Tensor &rhs, Tensor &out, int dim) {
        (void)(dim);
        this->reduce_with_broadcast(lhs, rhs, out);
    }

    void ElementWiseReduce::reduce_with_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
        this->reduce_with_broadcast(lhs, rhs, out);
    }

    int ElementWiseReduce::run(Stack &stack) {
        TS_AUTO_CHECK(stack.size() == 2);

        auto lhs = *stack.index(0);
        auto rhs = *stack.index(1);

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

        auto out_proto = Tensor::Prototype(lhs.dtype(), out_shape);

        auto memory_device = running_memory_device();

        lhs = lhs.view(memory_device).reshape(lhs_shape);    // do sync, and set default data to given device
        rhs = rhs.view(memory_device).reshape(rhs_shape);
        auto out = *stack.push(out_proto, memory_device);

        if (reduce_shape(lhs_shape, rhs_shape, out_shape)) {
            lhs = lhs.reshape(lhs_shape);
            rhs = rhs.reshape(rhs_shape);
            out = out.reshape(out_shape);
        }

        int dim;
        if (!do_broadcast) {
            reduce_with_same_shape(lhs, rhs, out);
        } else if (is_scalar(rhs_shape)) {
            reduce_with_scalar(lhs, rhs, out);
        } else if (is_scalar(lhs_shape)) {
            reduce_with_scalar_cross(lhs, rhs, out);
        } else if (is_bias(lhs_shape, rhs_shape, dim)) {
            reduce_with_bias(lhs, rhs, out, dim);
        } else if (is_bias(rhs_shape, lhs_shape, dim)) {
            reduce_with_bias_cross(lhs, rhs, out, dim);
        } else {
            reduce_with_broadcast(lhs, rhs, out);
        }

        return 1;
    }

    bool ElementWiseReduce::is_bias(Shape &lhs_shape, Shape &rhs_shape, int &dim) {
        auto count = std::accumulate(rhs_shape.begin(), rhs_shape.end(), 1, std::multiplies<int>());
        for (size_t i = 0; i < rhs_shape.size(); ++i) {
            if (rhs_shape[i] == count && lhs_shape[i] == count) {
                dim = int(i);
                return true;
            }
        }
        return false;
    }

    void ElementWiseReduce::reduce_with_bias_cross(const Tensor &lhs, const Tensor &rhs, Tensor &out, int dim) {
        (void)(dim);
        this->reduce_with_broadcast(lhs, rhs, out);
    }

    void ElementWiseReduce::reduce_with_scalar_cross(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
        this->reduce_with_broadcast(lhs, rhs, out);
    }

    bool ElementWiseReduce::reduce_shape(Shape &lhs, Shape &rhs, Shape &out) {
        bool reduced = false;
        bool lhs_ones = false;
        bool rhs_ones = false;
        bool lhs_equal_rhs = false;
        for (size_t i = 0; i < out.size(); ) {
            lhs_ones = lhs[i] == 1;
            rhs_ones = rhs[i] == 1;
            lhs_equal_rhs = lhs[i] == rhs[i];
            if (!(lhs_ones || rhs_ones || lhs_equal_rhs)) {
                ++i;
                continue;
            }
            auto j = i + 1;
            for (; j < out.size(); ++j) {
                if (lhs_ones) lhs_ones = lhs[j] == 1;
                if (rhs_ones) rhs_ones = rhs[j] == 1;
                if (lhs_equal_rhs) lhs_equal_rhs = lhs[j] == rhs[j];
                if (!(lhs_ones || rhs_ones || lhs_equal_rhs)) break;
            }
            if (j - i > 1) {
                auto a = std::accumulate(lhs.begin() + i, lhs.begin() + j, 1, std::multiplies<int32_t>());
                auto b = std::accumulate(rhs.begin() + i, rhs.begin() + j, 1, std::multiplies<int32_t>());
                auto c = std::accumulate(out.begin() + i, out.begin() + j, 1, std::multiplies<int32_t>());

                lhs.erase(lhs.begin() + i, lhs.begin() + j);
                rhs.erase(rhs.begin() + i, rhs.begin() + j);
                out.erase(out.begin() + i, out.begin() + j);

                lhs.insert(lhs.begin() + i, a);
                rhs.insert(rhs.begin() + i, b);
                out.insert(out.begin() + i, c);

                reduced = true;
                ++i;
                continue;
            } else {
                i = j + 1;
                continue;
            }
        }
        return reduced;
    }
}
