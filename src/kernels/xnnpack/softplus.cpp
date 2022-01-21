#include "softplus.h"
#include "backend/name.h"
#include "global/operator_factory.h"
#include <algorithm>
#include "kernels/common/simd.h"
#include "runtime/runtime.h"

namespace ts {
    namespace xnn {
        void Softplus::init() {
            supper::init();
            auto ctx = ctx::get<RuntimeContext>();
            m_pool = ctx->get_xnn_threadpool();
        }

        template<typename T>
        struct softplus_kernel_context {
            T* output_data;
            const T* input_data;
        };

        template<typename T>
        static inline void softplus_kernel(struct softplus_kernel_context<T> *context, size_t i) {
            context->output_data[i] = static_cast<float>(std::log(std::exp(context->input_data[i]) + 1.));
        }

        template<typename T>
        static void cpu_softplus_compute_run(const Tensor &x, Tensor &out, pthreadpool_t pool=nullptr) {
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            int count = out.count();

            std::memcpy(output_data, input_data, count * sizeof(T));

            int counts = out.count();
            for (int i = 0; i < counts; i++) {
                T val = *input_data;
                *output_data = std::log(std::exp(val) + 1.);
                output_data++;
            }
        }

        template<>
        void cpu_softplus_compute_run<float>(const Tensor &x, Tensor &out, pthreadpool_t pool) {
            auto input_data = x.data<float>();
            auto output_data = out.data<float>();
            int counts = out.count();

            softplus_kernel_context<float> context {output_data, input_data};
            pthreadpool_parallelize_1d(pool, (pthreadpool_task_1d_t) softplus_kernel<float>,
                                       (void**)&context, counts, 0);
        }

        void Softplus::active(const Tensor &x, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_softplus_compute_run<TYPE>(x, out, m_pool); break; }
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
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace xnn;
TS_REGISTER_OPERATOR(Softplus, ts::XNNPACK, "softplus")
