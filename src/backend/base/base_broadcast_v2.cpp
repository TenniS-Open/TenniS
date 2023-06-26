//
// Created by kier on 2019/10/21.
//

#include "backend/base/base_broadcast_v2.h"

#include "utils/assert.h"
#include "runtime/stack.h"

#include <numeric>
#include <core/tensor_builder.h>

namespace ts {
    namespace base {
        static inline void front_append_ones(Shape &shape, int count) {
            Shape ones(count, 1);
            shape.insert(shape.begin(), ones.begin(), ones.end());
        }

        static inline void front_append_ones(Shape &shape, size_t count) {
            Shape ones(count, 1);
            shape.insert(shape.begin(), ones.begin(), ones.end());
        }

        static inline void front_append_ones(std::vector<int> &shape, size_t count) {
            std::vector<int> ones(count, 1);
            shape.insert(shape.begin(), ones.begin(), ones.end());
        }

        void BroadcastV2::init() {
            supper::init();
        }

        int BroadcastV2::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto &x = stack[0];
            auto &shape = stack[1];

            auto x_shape = x.sizes().std();
            auto y_shape = tensor::array::to_int(shape);
            if (x_shape.size() < y_shape.size()) {
                front_append_ones(x_shape, y_shape.size() - x_shape.size());
            } else if (x_shape.size() != y_shape.size()) {
                front_append_ones(y_shape, x_shape.size() - y_shape.size());
            }
            auto z_shape = std::vector<int>();
            auto N = x_shape.size();
            for (decltype(N) i = 0; i < N; ++i) {
                if (x_shape[i] != 1 && y_shape[i] != 1 && x_shape[i] != y_shape[i]) {
                    z_shape.push_back(1);
                } else {
                    z_shape.push_back(std::max(x_shape[i], y_shape[i]));
                }
            }

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), z_shape);

            return 1;
        }

        bool BroadcastV2::is_scalar(const Shape &shape) {
            return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) == 1;
        }

        void BroadcastV2::broad_with_bias(const Tensor &x, Tensor &out, int dim) {
            (void) (dim);
            this->broadcast(x, out);
        }

        void BroadcastV2::broadcast_with_scalar(const Tensor &x, Tensor &out) {
            this->broadcast(x, out);
        }

        static bool reduce_shape(Shape &lhs, Shape &rhs, Shape &out) {
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

        int BroadcastV2::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto x = *stack.index(0);
            auto shape = *stack.index(1);

            auto x_shape = x.sizes();
            auto out_shape = Shape(tensor::array::to_int(shape));
            Shape target;

            bool do_broadcast = broadcast(x_shape, out_shape, target);
            if (out_shape.size() < target.size()) {
                front_append_ones(out_shape, target.size() - out_shape.size());
            }

            if (!do_broadcast) {
                stack.push(x.reshape(target));
                return 1;
            }

            auto out_proto = Tensor::Prototype(x.dtype(), target);

            auto memory_device = running_memory_device();

            x = x.view(memory_device).reshape(x_shape);    // do sync, and set default data to given device
            auto out = *stack.push(out_proto, memory_device);

            {
                auto y_shape = out_shape;
                if (reduce_shape(y_shape, x_shape, target)) {
                    x = x.reshape(x_shape);
                    out = out.reshape(target);
                }
            }

            int dim;
            if (is_scalar(x_shape)) {
                broadcast_with_scalar(x, out);
            } else if (is_bias(target, x_shape, dim)) {
                broad_with_bias(x, out, dim);
            } else {
                broadcast(x, out);
            }

            return 1;
        }

        bool BroadcastV2::is_bias(Shape &lhs_shape, Shape &rhs_shape, int &dim) {
            auto count = std::accumulate(rhs_shape.begin(), rhs_shape.end(), 1, std::multiplies<int>());
            for (size_t i = 0; i < rhs_shape.size(); ++i) {
                if (rhs_shape[i] == count && lhs_shape[i] == count) {
                    dim = int(i);
                    return true;
                }
            }
            return false;
        }

        bool BroadcastV2::broadcast(Shape &x, const Shape &shape, Shape &output) {
            auto y = shape;
            if (x.size() < y.size()) {
                front_append_ones(x, int(y.size() - x.size()));
            } else if (x.size() != y.size()) {
                front_append_ones(y, int(x.size() - y.size()));
            }
            if (x == y) {
                output = y;
                return false;
            }
            auto N = x.size();
            output.clear();
            for (size_t i = 0; i < N; ++i) {
                if (x[i] != y[i] && x[i] != 1 && y[i] != 1) {
                    TS_LOG_ERROR << "Can not broadcast " << to_string(x) << " to " << to_string(shape) << eject;
                }
                output.push_back(std::max(x[i], y[i]));
            }
            return true;
        }
    }
}
