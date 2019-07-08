//
// Created by kier on 19-6-17.
//

#include "runtime/operator.h"
#include "runtime/stack.h"
#include "global/operator_factory.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace cpu {
        class Expand : public Operator {
            int run(Stack &stack) final {
                TS_AUTO_CHECK(stack.size() == 2);
                auto &x = stack[0];
                auto dims = tensor::to_int(stack[1]);

                if (dims <= x.dims()) {
                    stack.push(x);
                } else {
                    auto new_shape = x.sizes();
                    while (new_shape.size() < size_t(dims)) {
                        new_shape.insert(new_shape.begin(), 1);
                    }
                    stack.push(x.reshape(new_shape));
                }

                return 1;
            }

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) final {
                TS_AUTO_CHECK(stack.size() == 2);
                auto &x = stack[0];
                auto dims = tensor::to_int(stack[1]);

                auto new_shape = x.sizes();
                while (new_shape.size() < size_t(dims)) {
                    new_shape.insert(new_shape.begin(), 1);
                }

                output.resize(1);
                output[0] = Tensor::Prototype(x.dtype(), new_shape);
                return 1;
            }
        };
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Expand, CPU, "_expand")