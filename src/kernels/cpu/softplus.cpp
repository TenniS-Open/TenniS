#include <kernels/cpu/softplus.h>
#include "backend/name.h"
#include "global/operator_factory.h"
#include <algorithm>

#include "kernels/common/simd.h"
#ifdef TS_USE_OPENMP
#include <kernels/common/openmp.h>
#endif

namespace ts {
    namespace cpu {
        template<typename T>
        static void cpu_softplus_compute_run(const Tensor &x, Tensor &out) {
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
        void cpu_softplus_compute_run<float>(const Tensor &x, Tensor &out) {
            auto input_data = x.data<float>();
            auto output_data = out.data<float>();
            int counts = out.count();
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int i = 0; i < counts; ++i) {
                *(output_data + i) = static_cast<float>(std::log(std::exp(*(input_data + i)) + 1.));
            }
        }

        void Softplus::active(const Tensor &x, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_softplus_compute_run<TYPE>(x, out); break; }
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
using namespace cpu;
TS_REGISTER_OPERATOR(Softplus, ts::CPU, "softplus")
