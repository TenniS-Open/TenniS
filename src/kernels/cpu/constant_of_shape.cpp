#include <kernels/cpu/constant_of_shape.h>
#include "backend/name.h"
#include "global/operator_factory.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace cpu {
        ConstantOfShape::ConstantOfShape() {
            field("value", OPTIONAL);
        }

        void ConstantOfShape::init() {
            supper::init();
        }

        int ConstantOfShape::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto x = stack[0];

            TS_AUTO_CHECK(x.dtype() == INT64);

            output.resize(1);
            if (has("value")) {
                output[0] = Tensor::Prototype(get("value").dtype(), x.sizes());
            } else {
                output[0] = Tensor::Prototype(FLOAT32, x.sizes());
            }

            return 1;
        }

        template<typename T>
        static void cpu_ConstantOfShape_compute_run(const Tensor &val, Tensor &out) {
            auto pout = out.data<T>();
            int count = out.count();

            auto value = val.data<T>();

            for (int i = 0; i < count; ++i) {
                pout[i] = *value;
            }
        }

        template<>
        void cpu_ConstantOfShape_compute_run<float>(const Tensor &val, Tensor &out) {
            auto pout = out.data<float>();
            int count = out.count();

            auto value = tensor::to_float(val);

            for (int i = 0; i < count; ++i) {
                *(pout + i) = value;
            }
        }

        int ConstantOfShape::run(Stack &stack) {
            std::vector<Tensor::Prototype> output_protos;

            infer(stack, output_protos);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto out = *stack.push(output_protos[0], memory_device);

            if (has("value")) {
                auto value = get("value");
                constant_of_shape(value, out);
            } else {
                auto value = tensor::build(FLOAT32, m_val);
                constant_of_shape(value, out);
            }

            return 1;
        }

        void ConstantOfShape::constant_of_shape(const Tensor &val, Tensor &out) {
            DTYPE dtype = val.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_ConstantOfShape_compute_run<TYPE>(val, out); break; }
                DECLARE_COMPUTE_RUN(BOOLEAN, uint8_t);
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
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(ConstantOfShape, ts::CPU, "constant_of_shape")
