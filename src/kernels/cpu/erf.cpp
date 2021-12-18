//
// Created by kier on 21-12-18.
//

#include "backend/base/base_activation.h"
#include "runtime/stack.h"
#include "global/operator_factory.h"

#include "kernels/cpu/operator_on_cpu.h"
#include "kernels/common/math.h"

#ifdef TS_USE_OPENMP
#include <kernels/common/openmp.h>
#endif

#include <cmath>

namespace ts {
    namespace cpu {
        template<typename T>
        static void cpu_erf_compute_run(const Tensor &x, Tensor &out, float epsilon=1e-5) {
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            int count = out.count();
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int i = 0; i < count; i++) {
                output_data[i] = erf(input_data[i]);
            }
        }

        class Erf : public OperatorOnCPU<base::Activation> {
        public:
            void active(const Tensor &x, Tensor &out) final {
                float epsilon=1e-5;
                DTYPE dtype = out.dtype();
                switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_erf_compute_run<TYPE>(x, out, epsilon); break; }
                    DECLARE_COMPUTE_RUN(FLOAT32, float);
                    DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                    default: {
                        TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                        break;
                    }
                }
            }
        };
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Erf, CPU, "erf")
