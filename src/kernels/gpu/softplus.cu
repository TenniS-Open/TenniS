#include <kernels/gpu/softplus.h>

#include "backend/name.h"
#include "global/operator_factory.h"
#include "global/fp16_operator_factory.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

#include "kernels/gpu/gpu_kernel.h"

#include "kernels/gpu/cudax_fp16_math.h"

namespace ts {
    namespace gpu {
        template<typename T>
        __global__ static void softplus_kernel(const T* input_data, T* output_data, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size) {
                output_data[index] = log(exp(input_data[index]) + T(1));
            }
        }

#ifdef TS_USE_CUDA_FP16
        template<>
        __global__ void softplus_kernel<half>(const half* input_data, half* output_data, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            half one(1.f);
            if (index < size) {
                output_data[index] = half(log(half(exp(input_data[index])) + one));
            }
        }
#endif

        template<typename T>
        static void cpu_softplus_compute_run(const Tensor &x, Tensor &out) {
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            int count = out.count();

            dim3 blockSize(CUDA_THREAD_NUM);
            dim3 gridSize(CUDA_BLOCK(count, blockSize.x));

            RUN_KERNEL(softplus_kernel<T>, gridSize, blockSize, input_data, output_data, count);
        }

        void Softplus::active(const Tensor &x, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
case DTYPE: { cpu_softplus_compute_run<TYPE>(x, out); break; }
#ifdef TS_USE_CUDA_FP16
                DECLARE_COMPUTE_RUN(FLOAT16, half);
#endif
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
using namespace gpu;
TS_REGISTER_OPERATOR(Softplus, ts::GPU, "softplus");
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(Softplus, ts::GPU, "softplus");
#endif
