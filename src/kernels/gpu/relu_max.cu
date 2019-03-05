#include <kernels/gpu/relu_max.h>

#include "backend/name.h"
#include "global/operator_factory.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#include <thrust/functional.h>

namespace ts {
    namespace gpu {
        template<typename T>
        __global__ static void relu_max_kernel(const T* input_data, T* output_data,T max, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            //thrust::maximum<T> mx;
            //thrust::minimum<T> mn;
            if (index < size)
            {
                //T val = input_data[index];
                //output_data[index] = mn(mx(val, T(0), max));
                T val = input_data[index];
                T max_temp = val > T(0) ? val : T(0);
                output_data[index] = max_temp < max ? max_temp : max;
            }
        }

        template<typename T>
        static void cpu_relu_max_compute_run(const Tensor &x, float max, Tensor &out) {
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            int count = out.count();

            T casted_max = T(max);
            
            dim3 blockSize(512);
            dim3 gridSize((count + blockSize.x - 1) / blockSize.x);

            relu_max_kernel<T> << <gridSize, blockSize >> > (input_data, output_data, casted_max, count);
        }

        void ReLUMax::relu_max(const Tensor &x, float max, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_relu_max_compute_run<TYPE>(x, max, out); break; }
                //DECLARE_COMPUTE_RUN(INT8, int8_t);
                //DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                //DECLARE_COMPUTE_RUN(INT16, int16_t);
                //DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                //DECLARE_COMPUTE_RUN(INT32, int32_t);
                //DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                //DECLARE_COMPUTE_RUN(INT64, int64_t);
                //DECLARE_COMPUTE_RUN(UINT64, uint64_t);
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
            default: {
                TS_LOG_ERROR << this->op() << " not support this data type: " << dtype << eject;
                break;
            }
            }
        }
    }
}

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(ReLUMax, ts::GPU, name::layer::relu_max())
