//
// Created by kier on 2019/7/26.
//

#include "backend/base/base_tile_v2.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        TileV2::TileV2() {
        }

        struct Repeats {
            Shape repeats;
            bool zeros = false;
        };

        static Repeats GetRepeats(const Tensor &data) {
            auto repeats = tensor::array::to_int(data);

            auto valid = true;
            auto zeros = false;
            for (auto repeat : repeats) {
                if (repeat < 0) {
                    valid = false;
                    break;
                }
                if (repeat == 0) {
                    zeros = true;
                }
            }

            if (!valid) {
                TS_LOG_ERROR << "Can not repeats " << to_string(repeats) << eject;
            }

            Repeats result;
            result.repeats = repeats;
            result.zeros = zeros;

            return result;
        }

        void TileV2::init() {
            supper::init();
        }

        static Shape infer_shape(Shape &x, Shape &repeats) {
            if (x.size() == repeats.size()) {
            } else if (x.size() > repeats.size()) {
                do {
                    repeats.insert(repeats.begin(), 1);
                } while (x.size() > repeats.size());
            } else{
                do {
                    x.insert(x.begin(), 1);
                } while (x.size() < repeats.size());
            }
            Shape y(x.size());
            for (size_t i = 0; i < x.size(); ++i) {
                y[i] = x[i] * repeats[i];
            }
            return y;
        }

        int TileV2::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto &x = stack[0];
            auto repeats = GetRepeats(stack[1]);

            auto local_x_shape = x.sizes();
            auto local_repeats = repeats.repeats;

            auto output_shape = infer_shape(local_x_shape, local_repeats);

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), output_shape);

            return 1;
        }

        int TileV2::run(ts::Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);
            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto repeats = GetRepeats(stack[1]);
            stack.pop(1);   // never used the tile.

            auto local_x_shape = x.sizes();
            auto local_repeats = repeats.repeats;

            auto output_shape = infer_shape(local_x_shape, local_repeats);

            auto &out = *stack.push(x.dtype(), output_shape, memory_device);

            if (repeats.zeros) return 1;

            x = x.reshape(local_x_shape);

            tile(x, local_repeats.std(), out);

            return 1;
        }
    }
}
