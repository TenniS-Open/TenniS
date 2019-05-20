#include <backend/base/base_topkv2.h>

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        Topkv2::Topkv2() {
            field(name::number,REQUIRED);
            field(name::sorted,OPTIONAL);
            m_sorted = 0;
        }

        void Topkv2::init() {
            supper::init();

            Tensor number_tensor =  get(name::number);
            TS_AUTO_CHECK(number_tensor.dtype() == INT32);
            m_number = tensor::to_int(number_tensor);

            if(has(name::sorted)) {
                Tensor sorted_tensor =  get(name::sorted);
                TS_AUTO_CHECK(sorted_tensor.dtype() == INT32);
                m_sorted = tensor::to_int(sorted_tensor);
            } 
        }


        int Topkv2::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);
             
            auto &x = stack[0];
            Shape x_shape = x.sizes();
            TS_AUTO_CHECK(x_shape[x_shape.size() - 1] >= m_number);
            x_shape[x_shape.size() - 1] = m_number;
            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), x_shape);

            return 1;
        }

        int Topkv2::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto memory_device = running_memory_device();
            auto x = stack[0].view(memory_device);

            Shape x_shape = x.sizes();
            TS_AUTO_CHECK(x_shape[x_shape.size() - 1] >= m_number);
            x_shape[x_shape.size() - 1] = m_number;
            auto output_proto = Tensor::Prototype(x.dtype(), x_shape);
            auto &out = *stack.push(output_proto, memory_device);

            topkv2(x, out);

            return 1;
        }
    }

}
