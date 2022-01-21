#include "exp.h"
#include <algorithm>

#include "backend/name.h"
#include "global/operator_factory.h"
#include "runtime/runtime.h"

namespace ts {
    namespace xnn {
        void Exp::init() {
            supper::init();
            auto ctx = ctx::get<RuntimeContext>();
            m_pool = ctx->get_xnn_threadpool();
        }

        template<typename T>
        struct exp_kernel_context {
            T* output_data;
            const T* input_data;
        };

        template<typename T>
        static inline void exp_kernel(struct exp_kernel_context<T> *context, size_t i) {
            context->output_data[i] = exp(context->input_data[i]);
        }

        template<typename T>
        static void cpu_exp_compute_run(const Tensor &x, Tensor &out, pthreadpool_t pool) {
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            int count = out.count();

            exp_kernel_context<T> context = {
                    output_data, input_data
            };
            pthreadpool_parallelize_1d(pool, (pthreadpool_task_1d_t) exp_kernel<T>,
                                       (void**)&context, count, 0);
        }


        void Exp::active(const Tensor &x, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_exp_compute_run<TYPE>(x, out, m_pool); break; }
/*
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, uint64_t);
*/
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
TS_REGISTER_OPERATOR(Exp, ts::XNNPACK, name::layer::exp())
