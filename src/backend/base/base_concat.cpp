//
// Created by kier on 2019/2/15.
//

#include <backend/base/base_concat.h>

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {

        Concat::Concat() {
            field(name::dim, REQUIRED);
        }

        void Concat::init() {
            supper::init();

            m_dim = tensor::to_int(this->get(name::dim));

            TS_AUTO_CHECK(m_dim >= 0);
        }

        int Concat::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            auto input_num = stack.size();

            TS_AUTO_CHECK(input_num != 0);
            // TS_AUTO_CHECK(stack.index(0)->dtype() == FLOAT32 || stack.index(0)->dtype() == FLOAT64);

            if (input_num == 1)
            {
                output.resize(1);
                output[0] = stack[0].proto();
                return 1;
            }

            auto dtype = stack.index(0)->dtype();

            for (int i = 1; i < input_num; i++)
            {
                TS_CHECK(stack.index(i)->dtype() == dtype) << "Can not concat on different data type!" << ts::eject;
            }

            Shape output_shape(stack.index(0)->sizes());

            TS_CHECK(m_dim < output_shape.size()) << "Concat dim should be less than input dim!" << ts::eject;

            auto num_dims = output_shape.size();
            int concat_dim_output_num = output_shape[m_dim];

            for (size_t i = 1; i < input_num; i++)
            {
                auto shape = stack.index(int(i))->sizes();
                TS_CHECK(shape.size() == num_dims) << "All inputs must have the same dims!" << ts::eject;

                for (int j = 0; j < shape.size(); j++)
                {
                    if (j == m_dim)
                        continue;
                    TS_CHECK(shape[j] == output_shape[j]) << "All inputs must have the same shape, except at concat_axis!" << ts::eject;
                }
                concat_dim_output_num += shape[m_dim];
            }

            output_shape[m_dim] = concat_dim_output_num;

            output.resize(1);
            output[0] = Tensor::Prototype(dtype, output_shape);

            return 1;
        }

        int Concat::run(Stack &stack) {
            std::vector<Tensor::Prototype> output_protos;
            infer(stack, output_protos);

            auto input_num = stack.size();

            auto memory_device = running_memory_device();

            std::vector<Tensor> x;
            for (size_t i = 0; i < input_num; ++i) {
                x.emplace_back(stack[i].view(memory_device));
            }

            Tensor out = *stack.push(output_protos[0], memory_device);

            concat(x, m_dim, out);

            return 1;
        }
    }
}
