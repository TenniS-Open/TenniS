#include "backend/base/base_scatter_nd.h"

namespace ts {
    namespace base {
        int ScatterND::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            auto &data = stack[0];
            output.resize(1);
            output[0] = Tensor::Prototype(data.dtype(), data.sizes());
            return 1;
        }

        int ScatterND::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 3);

            auto &data = stack[0];
            auto &indices = stack[1];
            auto &updates = stack[2];

            TS_CHECK(indices.dtype() == INT32) << "Indices should be int32, but got " << type_str(indices.dtype()) << eject;
            TS_CHECK(data.dtype() == updates.dtype()) << "Datatype must be same." << eject;

            std::vector<Tensor::Prototype> output;
            infer(stack, output);

            auto memory_device = running_memory_device();

            data = data.view(memory_device);
            indices = indices.view(memory_device);
            updates = updates.view(memory_device);

            Tensor &out = *stack.push(output[0], memory_device);

            scatter(data, indices, updates, out);

            return 1;
        }
    }
}