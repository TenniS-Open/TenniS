#include <kernels/gpu/prewhiten.h>
#include <algorithm>
#include "global/operator_factory.h"

#include "backend/name.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "core/device_context.h"
#include "utils/ctxmgr_lite.h"
#include "kernels/gpu/math_gpu.h"

namespace ts {
    namespace gpu {

        template<typename T>
        __global__ static void mean_kernel(const int N, T *x) {

            int index = blockDim.x * blockIdx.x + threadIdx.x;

            for (; index < 1; index += blockDim.x * gridDim.x) {
                x[index] /= N;
            }
        }

        template<typename T>
        __global__ static void dev_kernel(const int N, const T *x,T* mean, T * z) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            __shared__ T cache[CUDA_THREAD_NUM];

            int cache_index = threadIdx.x;
            T temp = T(0);
            for (; index < N; index += blockDim.x * gridDim.x) {
                T sub_tmp = x[index] - *mean;
                temp += sub_tmp * sub_tmp;
            }
            cache[cache_index] = temp;

            __syncthreads();

            unsigned int floor_pow = blockDim.x;
            if (floor_pow & (floor_pow - 1))
            {
                while (floor_pow & (floor_pow - 1))
                {
                    floor_pow &= (floor_pow - 1);
                }
                if (cache_index >= floor_pow)
                {
                    cache[cache_index - floor_pow] += cache[cache_index];
                }
                __syncthreads();
            }

            for (int i = floor_pow / 2; i > 0; i /= 2)
            {
                if (cache_index < i)
                {
                    cache[cache_index] += cache[cache_index + i];
                }
                __syncthreads();
            }

            if (cache_index == 0) {
                z[blockIdx.x] = cache[0];
            }
        }

        template<typename T>
        __global__ static void std_dev_kernel(const int N, T *x) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            for (; index < 1; index += gridDim.x * blockDim.x) {
                x[index] = sqrt(x[index] / N);
                x[index] = max(x[index], T(1 / sqrt(T(N))));
                x[index] = T(1) / x[index];
            }

        }

        template<typename T>
        __global__ static void prewhiten_kernel(const int N, T *x, T* mean,T * dev_rec) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            for (; index < N; index += gridDim.x * blockDim.x) {
                x[index] -= *mean;
                x[index] *= *dev_rec;
            }

        }

        template<typename T>
        static void gpu_pre_whiten_compute_run(const Tensor &x, Tensor &out) {
            auto output_shape = out.sizes();
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            int count = out.count();
            memcpy(output_data, out.device(), count * sizeof(T), input_data, x.device(), count * sizeof(T));
            //memcpy(output_data, input_data, count * sizeof(T));

            Shape mean_shape;
            mean_shape.resize(1);
            mean_shape[0] = 1;
            Tensor mean_tensor = Tensor(MemoryDevice(GPU), out.dtype(), mean_shape);
            T *mean = mean_tensor.data<T>();

            Shape std_dev_shape;
            std_dev_shape.resize(1);
            std_dev_shape[0] = 1;
            Tensor std_dev_tensor = Tensor(MemoryDevice(GPU), out.dtype(), std_dev_shape);
            T *std_dev = std_dev_tensor.data<T>();

            T *at = nullptr;

            // fot batch
            int batch = x.size(0);
            count /= batch;
            auto batch_outout_data = output_data;

            for (int n = 0; n < batch; ++n) {
                at = batch_outout_data;
                math<T>::sum(count, at,mean);
                mean_kernel<T> << <1,1 >> > (count,mean);

                at = batch_outout_data;
                int grid_size = CUDA_BLOCK(count, CUDA_THREAD_NUM);
                int block_size = CUDA_THREAD_NUM;
                Shape dev_shape;
                dev_shape.resize(1);
                dev_shape[0] = 1;
                Tensor dev_tensor = Tensor(MemoryDevice(GPU), out.dtype(), dev_shape);
                T* dev_buffer = dev_tensor.data<T>();
                //T* tmp_dev_out = (T*)gpu_allocator(device_id, grid_size * sizeof(T), nullptr, 0);
                dev_kernel<T> << < grid_size, block_size >> > (count, at, mean, dev_buffer);
                math<T>::sum(grid_size, dev_buffer, std_dev);
                std_dev_kernel<T> << <1,1 >> > (count, std_dev);

                at = batch_outout_data;
                prewhiten_kernel<T> << <grid_size,block_size >> > (count,at,mean, std_dev);

                batch_outout_data += count;
            }
        }

        void PreWhiten::prewhiten(const Tensor &x, Tensor &out) {
            auto dtype = out.dtype();
            switch (dtype) {
#define DECLARE_TYPE_AND_RUN(DTYPE, TYPE) \
				case DTYPE: { gpu_pre_whiten_compute_run<TYPE>(x, out); break; }
                //DECLARE_TYPE_AND_RUN(INT8, int8_t);
                //DECLARE_TYPE_AND_RUN(UINT8, uint8_t);
                //DECLARE_TYPE_AND_RUN(INT16, int16_t);
                //DECLARE_TYPE_AND_RUN(UINT16, uint16_t);
                //DECLARE_TYPE_AND_RUN(INT32, int32_t);
                //DECLARE_TYPE_AND_RUN(UINT32, uint32_t);
                //DECLARE_TYPE_AND_RUN(INT64, int64_t);
                //DECLARE_TYPE_AND_RUN(UINT64, uint64_t);
                DECLARE_TYPE_AND_RUN(FLOAT32, float);
                DECLARE_TYPE_AND_RUN(FLOAT64, double);
#undef DECLARE_TYPE_AND_RUN
            default: {
                TS_LOG_ERROR << "pre_whiten not support this data type: " << type_str(dtype) << eject;
                break;
            }
            }
        }
    }
}

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(PreWhiten, GPU, "prewhiten")
