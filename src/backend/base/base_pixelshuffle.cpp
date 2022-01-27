//
// Created by sen on 2022/1/25.
//

#include "backend/base/base_pixelshuffle.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        PixelShuffle::PixelShuffle() {
            field("upscale_factor", REQUIRED, tensor::from<int32_t>(1));
            field("mode", OPTIONAL, tensor::from("DCR"));
        }

        void PixelShuffle::init() {
            supper::init();
            m_upscale_factor = tensor::to_int(get("upscale_factor"));
            m_mode = tensor::to_string(get("mode"));
        }

        int PixelShuffle::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            auto x_shape = stack[0].sizes();
            TS_CHECK(x_shape.size() == 4);

            x_shape[1] /= m_upscale_factor * m_upscale_factor;
            x_shape[2] *= m_upscale_factor;
            x_shape[3] *= m_upscale_factor;

            output.emplace_back(TensorPrototype(stack[0].dtype(), x_shape));
            return 1;
        }

        int PixelShuffle::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto x = *stack.index(0);

            std::vector<Tensor::Prototype> output;
            infer(stack, output);

            auto memory_device = running_memory_device();
            x = x.view(memory_device);

            Tensor &out = *stack.push(output[0], memory_device);
            pixel_shuffle(x, out, m_upscale_factor, m_mode);

            return 1;
        }
    }
}
