//
// Created by kier on 20-9-10.
//

#include "runtime/operator.h"
#include "global/operator_factory.h"
#include "core/tensor_builder.h"

#include "backend/base/base_pow.h"
#include "kernels/cpu//operator_on_cpu.h"

#ifdef TS_USE_OPENMP

#include <kernels/common/openmp.h>

#endif

namespace ts {
    namespace cpu {
        template<typename T>
        static void cpu_pow_compute_run(const Tensor &x, float y, Tensor &out) {
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            int count = out.count();

//            std::memcpy(output_data, input_data, count * sizeof(T));
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int i = 0; i < count; i++) {
                output_data[i] = std::pow(input_data[i], T(y));
            }
        }

        class Pow : public OperatorOnCPU<base::Pow> {
        public:
            using self = Pow;
            using supper = OperatorOnCPU<base::Pow>;

            void pow(const Tensor &x, float y, Tensor &out) override {
                // Notice: the all tensors' memory device are CPU, as given in running_memory_device
                DTYPE dtype = out.dtype();
                switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_pow_compute_run<TYPE>(x, y, out); break; }
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
                        TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype)
                                     << eject;
                        break;
                    }
                }
            }
        };
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Pow, ts::CPU, "pow")
