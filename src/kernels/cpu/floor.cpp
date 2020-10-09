//
// Created by kier on 20-9-10.
//

#include "runtime/operator.h"
#include "runtime/stack.h"
#include "global/operator_factory.h"
#include "core/tensor_builder.h"
#include <algorithm>

#include <algorithm>

namespace ts {
    namespace cpu {
        namespace {
            template<typename T>
            inline typename std::enable_if<std::is_integral<T>::value, int>::type
            compute_floor(Stack &stack, const Tensor &x) {
                return 1;
            }

            template<typename T>
            inline typename std::enable_if<std::is_floating_point<T>::value, int>::type
            compute_floor(Stack &stack, const Tensor &x) {
                auto cpu_x = x.view(CPU);
                auto count = x.count();
                auto &y = *stack.push(x.proto(), CPU);
                auto x_data = cpu_x.data<T>();
                auto y_data = y.data<T>();
                for (decltype(count) i = 0; i < count; ++i) {
                    y_data[i] = std::floor(x_data[i]);
                }
                return 1;
            }

        }

        class Floor : public Operator {
        public:
            Floor() = default;

            void init() final {}

            int run(Stack &stack) final {
                TS_AUTO_ASSERT(stack.size() == 1);
                auto &x = stack[0];

                DTYPE dtype = x.dtype();
                switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { compute_floor<TYPE>(stack, x); break; }
                    DECLARE_COMPUTE_RUN(INT8, int8_t);
                    DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                    DECLARE_COMPUTE_RUN(INT16, int16_t);
                    DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                    DECLARE_COMPUTE_RUN(INT32, int32_t);
                    DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                    DECLARE_COMPUTE_RUN(INT64, int64_t);
                    DECLARE_COMPUTE_RUN(UINT64, uint64_t);
                    DECLARE_COMPUTE_RUN(FLOAT32, float);
                    DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                    default: {
                        TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype)
                                     << eject;
                        break;
                    }
                }
                return 1;
            }

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) final {
                TS_AUTO_ASSERT(stack.size() == 1);
                auto &x = stack[0];

                output.resize(1);
                output[0] = Tensor::Prototype(x.dtype(), x.sizes());
                return 1;
            }
        };
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Floor, CPU, "floor")