#include <kernels/gpu/constant_of_shape.h>
#include <global/operator_factory.h>
#include <global/fp16_operator_factory.h>
#include <backend/name.h>
#include "core/tensor_builder.h"
#include <kernels/gpu/cudax_fp16_math.h>

namespace ts {
    namespace gpu {
        ConstantOfShape::ConstantOfShape() {
            field("value", OPTIONAL);
        }

        void ConstantOfShape::init() {
            supper::init();
            if (has("value")) {
                m_tensor = get("value");
            } else {
                m_tensor = tensor::build(FLOAT32, 0.f);
            }
        }

        int ConstantOfShape::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto x = stack[0];

            output.resize(1);
            output[0] = Tensor::Prototype(m_tensor.dtype(), x.sizes());

            return 1;
        }

        int ConstantOfShape::run(Stack &stack) {
            std::vector<Tensor::Prototype> output;

            infer(stack, output);

            auto memory_device = running_memory_device();

            auto out = *stack.push(output[0], memory_device);

            constant_of_shape(m_tensor, out);

            return 1;
        }

        template<typename T>
        static void gpu_ConstantOfShape_compute_run(const Tensor &val, Tensor &out) {
            memset(out.data(), out.device(), out.count() * out.proto().type_bytes(),
                   val.data(), val.device(), val.count() * val.proto().type_bytes());
        }

        void ConstantOfShape::constant_of_shape(const Tensor &val, Tensor &out) {
            DTYPE dtype = val.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_ConstantOfShape_compute_run<TYPE>(val, out); break; }
                DECLARE_COMPUTE_RUN(BOOLEAN, uint8_t);
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, uint64_t);
#ifdef TS_USE_CUDA_FP16
                DECLARE_COMPUTE_RUN(FLOAT16, half);
#endif
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
using namespace gpu;
TS_REGISTER_OPERATOR(ConstantOfShape, GPU, "constant_of_shape")
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(ConstantOfShape, GPU, "constant_of_shape")
#endif
