#include "backend/base/base_pow.h"

#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        Pow::Pow() {
            field("y", REQUIRED);
        }

        void Pow::init() {
            supper::init();

            Tensor y_tensor = tensor::cast(FLOAT32, get("y"));
            m_y = tensor::to_float(y_tensor);

        }

        int Pow::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            output.resize(1);
            output[0] = stack[0].proto();

            return 1;
        }

        int Pow::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto out = *stack.push(x.proto(), memory_device);

            pow(x, m_y, out);

            return 1;
        }
    }
}