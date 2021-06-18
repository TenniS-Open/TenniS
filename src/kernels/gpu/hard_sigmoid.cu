#include <backend/base/base_hard_sigmoid.h>
#include "kernels/gpu/operator_on_gpu.h"

#include "backend/name.h"
#include "global/operator_factory.h"
#include "global/fp16_operator_factory.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

#include "kernels/gpu/gpu_kernel.h"

namespace ts {
    namespace gpu {
        class HardSigmoid : public OperatorOnGPU<base::HardSigmoid> {
        public:
            using self = HardSigmoid;
            using supper = OperatorOnGPU<base::HardSigmoid>;

            void hard_sigmoid(const Tensor &x, float alpha, float beta, Tensor &out) override;
        };

        template<typename T>
        __global__ static void hard_sigmoid_kernel(const T* input_data, T* output_data,
                                                   T alpha, T beta, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size)
            {
                T val = input_data[index];
                output_data[index] = max(0., min(1., alpha * val + beta));
            }
        }

#ifdef TS_USE_CUDA_FP16
        template<>
        __global__ void hard_sigmoid_kernel<half>(const half* input_data, half* output_data,
                                                  half alpha, half beta, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            half zero(0.f);
            half one(1.f);
            if (index < size)
            {
                half val = input_data[index];
                half min_partial = alpha * val + beta < one ? alpha * val + beta : one;
                half max_partial = min_partial > zero ? min_partial : zero;
                output_data[index] = max_partial;
            }
        }
#endif

        template<typename T>
        static void gpu_hard_sigmoid_compute_run(const Tensor &x, float alpha, float beta, Tensor &out) {
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            int count = out.count();

            T casted_alpha = T(alpha);
            T casted_beta = T(beta);

            dim3 blockSize(CUDA_THREAD_NUM);
            dim3 gridSize(CUDA_BLOCK(count, blockSize.x));

            RUN_KERNEL(hard_sigmoid_kernel<T>, gridSize, blockSize, input_data, output_data, casted_alpha, casted_beta, count);
        }

        void HardSigmoid::hard_sigmoid(const Tensor &x, float alpha, float beta, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_hard_sigmoid_compute_run<TYPE>(x, alpha, beta, out); break; }
                //DECLARE_COMPUTE_RUN(INT8, int8_t);
                //DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                //DECLARE_COMPUTE_RUN(INT16, int16_t);
                //DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                //DECLARE_COMPUTE_RUN(INT32, int32_t);
                //DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                //DECLARE_COMPUTE_RUN(INT64, int64_t);
                //DECLARE_COMPUTE_RUN(UINT64, uint64_t);
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
TS_REGISTER_OPERATOR(HardSigmoid, ts::GPU, "hard_sigmoid")
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(HardSigmoid, ts::GPU, "hard_sigmoid")
#endif
