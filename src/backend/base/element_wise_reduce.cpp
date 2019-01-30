//
// Created by kier on 2019/1/28.
//

#include <backend/base/element_wise_reduce.h>

#include "backend/base/element_wise_reduce.h"

#include "utils/assert.h"
#include "runtime/stack.h"

#include <numeric>

namespace ts {
    static inline void front_append_ones(Shape &shape, int count) {
        Shape ones(count, 1);
        shape.insert(shape.begin(), ones.begin(), ones.end());
    }

    bool ElementWiseReduce::reduce(Shape &_lhs_shape, Shape &_rhs_shape, Shape &_out_shape, bool broadcast) {

        if (_lhs_shape == _rhs_shape) {
            _out_shape = _lhs_shape;
            return false;
        }

        if (!broadcast) {
            TS_LOG_ERROR << "Can not reduce shape: " << to_string(_lhs_shape) << " vs. " << to_string(_rhs_shape) << eject;
        }

        TS_AUTO_CHECK(!_lhs_shape.empty());
        TS_AUTO_CHECK(!_rhs_shape.empty());

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
                    TS_LOG_ERROR << "Can not reduce shape: " << to_string(_lhs_shape) << " vs. " << to_string(_rhs_shape);
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

    int ElementWiseReduce::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
        TS_AUTO_CHECK(stack.size() == 2);

        auto lhs = *stack.index(0);
        auto rhs = *stack.index(1);

        TS_AUTO_CHECK(lhs.dtype() == rhs.dtype());

        auto lhs_shape = lhs.sizes();
        auto rhs_shape = rhs.sizes();
        Shape out_shape;

        bool do_broadcast = reduce(lhs_shape, rhs_shape, out_shape, true);
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

        TS_AUTO_CHECK(lhs.dtype() == rhs.dtype());

        auto lhs_shape = lhs.sizes();
        auto rhs_shape = rhs.sizes();
        Shape out_shape;

        bool do_broadcast = reduce(lhs_shape, rhs_shape, out_shape, true);

        auto out_proto = Tensor::Prototype(lhs.dtype(), out_shape);

        auto memory_device = running_memory_device();

        lhs = lhs.view(memory_device).reshape(lhs_shape);    // do sync, and set default data to given device
        rhs = rhs.view(memory_device).reshape(rhs_shape);
        auto out = *stack.push(out_proto, memory_device);

        int dim;
        if (!do_broadcast) {
            reduce_with_same_shape(lhs, rhs, out);
        } else if (is_scalar(rhs_shape)) {
            reduce_with_scalar(lhs, rhs, out);
        } else if (is_scalar(lhs_shape)) {
            reduce_with_scalar(rhs, lhs, out);
        } else if (is_bias(lhs_shape, rhs_shape, dim)) {
            reduce_with_bias(lhs, rhs, out, dim);
        } else if (is_bias(rhs_shape, lhs_shape, dim)) {
            reduce_with_bias(rhs, lhs, out, dim);
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
}
