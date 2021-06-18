//
// Created by sen on 2021/6/15.
//

#include <backend/base/base_scatter_nd.h>
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
        class ScatterND : public OperatorOnGPU<base::ScatterND> {
        public:
            using self = ScatterND;
            using supper = OperatorOnGPU<base::ScatterND>;

            void scatter(const Tensor &data, const Tensor &indices, Tensor &updates, Tensor &out) override;
        };


        template<typename T>
        __global__ static void scatter_kernel(const T* indices, const T* updates, T* out, int updates_slice_step, int bytes, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size) return;

            int indices_data = indices[index];
            auto dst_ptr = out + indices_data * updates_slice_step;
            auto src_ptr = updates + index * updates_slice_step;
            ::memcpy((void *)dst_ptr, (void *)src_ptr, updates_slice_step * bytes);
        }


        template<typename T>
        static void gpu_scatter_compute_run(const Tensor &data, const Tensor &indices, Tensor &updates, Tensor &out) {
            const T *pdata = data.data<T>();
            const T *pindices = indices.data<T>();
            T *pupdates = updates.data<T>();
            T *pout = out.data<T>();
            int count = updates.count();

            memcpy((void*)pout, out.device(), out.count() * sizeof(T),
                   (void*)pdata, data.device(), data.count() * sizeof(T));

            dim3 blockSize(CUDA_THREAD_NUM);
            dim3 gridSize(CUDA_BLOCK(count, blockSize.x));

            auto update_indices = indices.sizes();
            int update_index_cnt = update_indices[0];
            // the length of step is equal
            int updates_slice_step = updates.slice(0).count();
            auto bytes = out.proto().type_bytes();

            RUN_KERNEL(scatter_kernel<T>, gridSize, blockSize, pindices, pupdates, pout, updates_slice_step, bytes, update_index_cnt);

        }

        void ScatterND::scatter(const Tensor &data, const Tensor &indices, Tensor &updates, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_scatter_compute_run<TYPE>(data, indices, updates, out); break; }
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, uint64_t);
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
TS_REGISTER_OPERATOR(ScatterND, GPU, "scatter_nd")
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(ScatterND, GPU, "scatter_nd")
#endif