#include "backend/base/base_where.h"
#include "frontend/intime.h"

namespace ts {
    namespace base {
        static inline void front_append_ones(Shape &shape, int count) {
            Shape ones(count, 1);
            shape.insert(shape.begin(), ones.begin(), ones.end());
        }

        bool Where::reduce(Operator *op, Shape &_cond_shape, Shape &_lhs_shape, Shape &_rhs_shape,
                           Shape &_out_shape, bool broadcast) {
            if (_cond_shape == _lhs_shape && _cond_shape == _rhs_shape) {
                _out_shape = _cond_shape;
                return false;
            }

            if (!broadcast) {
                TS_LOG_ERROR << "[" << op->op() << ":" << op->name() << "] Can not reduce shape: "
                             << to_string(_cond_shape) << " vs. " << to_string(_lhs_shape) << " vs. "
                             << to_string(_rhs_shape) << eject;
            }

            auto cond_shape = _cond_shape;
            auto lhs_shape = _lhs_shape;
            auto rhs_shape = _rhs_shape;

            auto cond_dim = int(cond_shape.size());
            auto lhs_dim = int(lhs_shape.size());
            auto rhs_dim = int(rhs_shape.size());

            auto max_dim = std::max(std::max(cond_dim, lhs_dim), rhs_dim);

            if (cond_shape.size() < max_dim) {
                front_append_ones(cond_shape, int(max_dim - cond_dim));
            }
            if (lhs_shape.size() < max_dim) {
                front_append_ones(lhs_shape, int(max_dim - lhs_dim));
            }
            if (rhs_shape.size() < max_dim) {
                front_append_ones(rhs_shape, int(max_dim - rhs_dim));
            }

            auto dims = cond_shape.size();

            Shape out_shape(dims);
            bool do_broadcast = false;

            for (size_t i = 0; i < dims; ++i) {
                int size = cond_shape[i];
                if (!(cond_shape[i] == lhs_shape[i] && cond_shape[i] == rhs_shape[i])) {
                    // any two of shape[i] not equal to one, throw error
                    if ((cond_shape[i] != 1 && lhs_shape[i] != 1 && cond_shape[i] != lhs_shape[i]) ||
                        (cond_shape[i] != 1 && rhs_shape[i] != 1 && cond_shape[i] != rhs_shape[i]) ||
                        (lhs_shape[i] != 1 && rhs_shape[i] != 1 && lhs_shape[i] != rhs_shape[i])) {
                        TS_LOG_ERROR << "[" << op->op() << ":" << op->name() << "] Can not reduce shape: "
                                     << to_string(_cond_shape) << " vs. " << to_string(_lhs_shape) << " vs. "
                                     << to_string(_rhs_shape) << eject;
                    }
                    do_broadcast = true;
                    size = std::max((std::max(cond_shape[i], lhs_shape[i])), rhs_shape[i]);
                }
                out_shape[i] = size;
            }

            _cond_shape = cond_shape;
            _lhs_shape = lhs_shape;
            _rhs_shape = rhs_shape;
            _out_shape = out_shape;

            return do_broadcast;
        }

        int Where::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 3);

            auto cond = *stack.index(0);
            auto lhs = *stack.index(1);
            auto rhs = *stack.index(2);

            if (lhs.dtype() != rhs.dtype()) {
                TS_LOG_ERROR << "[" << this->op() << ":" << this->name() << "] Can not reduce mismatch type: "
                             << type_str(cond.dtype()) << " vs. "
                             << type_str(lhs.dtype()) << " vs. "
                             << type_str(rhs.dtype()) << eject;
            }

            auto cond_shape = cond.sizes();
            auto lhs_shape = lhs.sizes();
            auto rhs_shape = rhs.sizes();
            Shape out_shape;

            bool do_broadcast = reduce(this, cond_shape, lhs_shape, rhs_shape, out_shape, true);
            TS_UNUSED(do_broadcast);

            output.resize(1);
            output[0] = Tensor::Prototype(lhs.dtype(), out_shape);

            return 1;
        }

        int Where::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 3);

            auto cond = *stack.index(0);
            auto lhs = *stack.index(1);
            auto rhs = *stack.index(2);

            std::vector<Tensor::Prototype> output;
            infer(stack, output);

            cond = intime::cast(cond, BOOLEAN);

            auto cond_shape = cond.sizes();
            auto lhs_shape = lhs.sizes();
            auto rhs_shape = rhs.sizes();

            Shape out_shape;

            bool do_broadcast = reduce(this, cond_shape, lhs_shape, rhs_shape, out_shape, true);

            auto memory_device = running_memory_device();

            cond = cond.view(memory_device).reshape(cond_shape);    // do sync, and set default data to given device
            lhs = lhs.view(memory_device).reshape(lhs_shape);
            rhs = rhs.view(memory_device).reshape(rhs_shape);

            Tensor &out = *stack.push(output[0], memory_device);

            if (!do_broadcast) {
                reduce_with_same_shape(cond, lhs, rhs, out);
            } else {
                reduce_with_broadcast(cond, lhs, rhs, out);
            }

            return 1;
        }
    }
}