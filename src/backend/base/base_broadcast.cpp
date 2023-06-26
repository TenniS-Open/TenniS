//
// Created by kier on 2019/10/17.
//

#include "backend/base/base_broadcast.h"
#include "core/tensor_builder.h"

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

        int Broadcast::infer(ts::Stack &stack, std::vector<Tensor::Prototype> &output) {
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

        static bool can_broadcast(const Shape &x, const Shape &y, std::vector<int> &z) {
            // if (x.size() != y.size()) return false;
            auto xt = x;
            auto yt = y;
            if (xt.size() < yt.size()) {
                do {
                    xt.insert(xt.begin(), 1);
                } while (xt.size() < yt.size());
            } else if (xt.size() > yt.size()) {
                do {
                    yt.insert(xt.begin(), 1);
                } while (xt.size() > yt.size());
            }
            z.clear();
            auto N = x.size();
            for (size_t i = 0; i < N; ++i) {
                if (x[i] != 1 && y[i] != 1 && x[i] != y[i]) return false;
                z.push_back(std::max(x[i], y[i]));
            }
            return true;
        }

        int Broadcast::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto shape = tensor::array::to_int(stack[1]);

            std::vector<int> target;

            if (!can_broadcast(x.sizes(), shape, target)) {
                TS_LOG_ERROR << "Can not broadcast x.shape="
                             << to_string(x.sizes()) << " to " << to_string(shape) << eject;
            }

            auto &out = *stack.push(x.dtype(), shape, memory_device);
            auto x_shape = x.sizes();
            while (x_shape.size() < target.size()) {
                x_shape.insert(x_shape.begin(), 1);
            }

            broadcast(x.reshape(x_shape), target, out);

            return 1;
        }
    }
}
