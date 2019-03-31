#include <kernels/gpu/pad.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

#include <core/memory.h>
#include <numeric>

#include "kernels/gpu/gpu_helper.h"

namespace ts {
    namespace gpu {
        template <typename T>
        static __global__ void pad_gpu_kernel(int count, const int *padding, T padding_value, const T *in, T *out, GpuHypeShape in_shape, GpuHypeShape out_shape) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= count) return;

            int out_index = index;
            int in_index = 0;

            auto out_weight_it = out_shape.weights + 1;
            auto in_weight_it = in_shape.weights + 1;
            /* ============================================ */
            auto in_shape_it = in_shape.shape;
            auto padding_it = padding;
            /* -------------------------------------------- */

            for (int times = out_shape.dims - 1; times; --times) {
                auto coord = index / *out_weight_it;
                /* ============================================ */
                coord -= *padding_it;
                if (coord < 0 || coord >= *in_shape_it) {
                    out[out_index] = padding_value;
                    return;
                }
                ++in_shape_it;
                padding_it += 2;
                /* -------------------------------------------- */
                in_index += coord * *in_weight_it;
                index %= *out_weight_it;
                ++out_weight_it;
                ++in_weight_it;
            }
            auto coord = index;
            /* ============================================ */
            coord -= *padding_it;
            if (coord < 0 || coord >= *in_shape_it) {
                out[out_index] = padding_value;
                return;
            }
            /* -------------------------------------------- */
            in_index += coord;

            /* ++++++++++++++++++++++++++++++++++++++++++++ */
            out[out_index] = in[in_index];
        }

        template <typename T>
        static inline void pad_gpu_compute_run(const Tensor &x, const std::vector<std::array<int, 2>> &padding, float padding_value, Tensor &out) {
            int *gpu_padding = nullptr;
            auto gpu_memory = MakeGPUHypeShape(out.device(), {x.sizes(), out.sizes()},
                                               {{(void *) (padding.data()), sizeof(int) * int(padding.size()) * 2}},
                                               {(void **) (&gpu_padding)});
            auto &gpu_in_shape = gpu_memory.second[0];
            auto &gpu_out_shape = gpu_memory.second[1];
            auto in_data = x.data<T>();
            auto out_data = out.data<T>();
            auto count = out.count();
            pad_gpu_kernel<T> << < CUDA_BLOCK(count, CUDA_THREAD_NUM), CUDA_THREAD_NUM >> > (count, gpu_padding, T(padding_value), in_data, out_data, gpu_in_shape, gpu_out_shape);
        }

        void PadOnGPU::pad(const Tensor &x, const std::vector<std::array<int, 2>> &padding, float padding_value, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch(dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { pad_gpu_compute_run<TYPE>(x, padding, padding_value, out); break; }
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << "pad not support this data type: " << dtype << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(PadOnGPU, GPU, name::layer::pad())
