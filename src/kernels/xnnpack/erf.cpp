//
// Created by sen on 2022/1/21.
//

#include "erf.h"
#include "utils/ctxmgr_lite.h"
#include "kernels/xnnpack/threadpool.h"
#include "backend/name.h"
#include "core/device.h"
#include "global/operator_factory.h"
#include "runtime/runtime.h"

namespace ts {
    namespace xnn {
        void Erf::init() {
            supper::init();
            auto ctx = ctx::get<RuntimeContext>();
            m_pool = ctx->get_xnn_threadpool();
        }

        template<typename T>
        struct erf_kernel_context {
            T* output_data;
            const T* input_data;
        };

        template<typename T>
        static inline void erf_kernel(struct erf_kernel_context<T> *context, size_t i) {
            context->output_data[i] = erf(context->input_data[i]);
        }

        template<typename T>
        static void cpu_erf_compute_run(const Tensor &x, Tensor &out, pthreadpool_t pool, float epsilon=1e-5) {
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            int count = out.count();

            erf_kernel_context<T> kernel_context {output_data, input_data};
            pthreadpool_parallelize_1d(pool, (pthreadpool_task_1d_t) erf_kernel<T>,
                                       (void**)&kernel_context, count, 0);
        }

        void Erf::active(const Tensor &x, Tensor &out) {
            float epsilon=1e-5;
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_erf_compute_run<TYPE>(x, out, m_pool, epsilon); break; }
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
TS_REGISTER_OPERATOR(Erf, ts::XNNPACK, "erf")
