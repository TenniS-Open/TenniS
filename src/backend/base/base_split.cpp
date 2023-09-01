#include "backend/base/base_split.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        Split::Split() {
            field(name::axis, REQUIRED);
        }

        void Split::init() {
            supper::init();

            Tensor dim_tensor = tensor::cast(INT32, get(name::axis));
            m_dim = tensor::to_int(dim_tensor);

        }

        static std::vector<Tensor::Prototype> infer_split(const Tensor &x, const std::vector<int> &split, int dim) {
            Shape x_shape = x.sizes();
            if(dim < 0) {
                dim += int(x_shape.size());
            }

            TS_AUTO_CHECK((dim >= 0) && (dim < int(x_shape.size())));

            std::vector<Tensor::Prototype> shapes;
            for (auto size : split) {
                auto shape = x_shape;
                shape[dim] = size;
                shapes.emplace_back(x.dtype(), shape);
            }

            return shapes;
        }

        int Split::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto x = stack[0];
            auto split = tensor::array::to_int(stack[1]);

            output = infer_split(x, split, m_dim);

            return 1;
        }

        int Split::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto split = tensor::array::to_int(stack[1]);

            auto dim = m_dim;
            Shape x_shape = x.sizes();
            if(dim < 0) {
                dim += int(x_shape.size());
            }

            TS_AUTO_CHECK((dim >= 0) && (dim < int(x_shape.size())));

            auto output_protos = infer_split(x, split, dim);
            std::vector<Tensor> outputs;
            outputs.reserve(output_protos.size());
            for (auto &proto : output_protos) {
                outputs.push_back(stack.make(proto, memory_device));
            }

            this->split(x, split, dim, outputs);

            Tensor out;
            out.pack(outputs);
            stack.push(out);

            return 1;
        }
    }
}