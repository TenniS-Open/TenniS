#include <kernels/gpu/cast.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <utils/assert.h>
#include <core/device.h>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>

/////////////////////////////////////////////////
namespace ts {
    namespace gpu {
    template<typename T_IN, typename T_OUT>
    static __global__ void gpu_cast_kernel(T_OUT * dst, const T_IN * src, int size) {
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        if (index < size) {
            dst[index] = static_cast<T_OUT>(src[index]);
        }
    }

    template<typename T_IN, typename T_OUT>
    static void gpu_cast_compute_run_template(const Tensor &x, Tensor &out) {

        const T_IN *psrc = x.data<T_IN>();
        T_OUT *pdst = out.data<T_OUT>();

        if(x.dtype() == out.dtype()) {
            memcpy((void*)pdst, out.device(), x.count() * sizeof(T_IN),
                   (void*)psrc, x.device(), x.count() * sizeof(T_IN));

            return;
        }

        gpu_cast_kernel<T_IN, T_OUT> <<< CUDA_BLOCK(x.count(), CUDA_THREAD_NUM), CUDA_THREAD_NUM >>> (pdst, psrc, x.count()); 

    }

    template<typename T_IN>
    static void gpu_cast_compute_run(const Tensor &x, DTYPE to_type, Tensor &out) {
        switch (to_type) {
#define DECLARE_COMPUTE_RUN_TEMPLATE(DTYPE, TYPE) \
        case DTYPE: { gpu_cast_compute_run_template<T_IN, TYPE>(x, out); break; }
            DECLARE_COMPUTE_RUN_TEMPLATE(INT8, int8_t);
            DECLARE_COMPUTE_RUN_TEMPLATE(UINT8, uint8_t);
            DECLARE_COMPUTE_RUN_TEMPLATE(INT16, int16_t);
            DECLARE_COMPUTE_RUN_TEMPLATE(UINT16, uint16_t);
            DECLARE_COMPUTE_RUN_TEMPLATE(INT32, int32_t);
            DECLARE_COMPUTE_RUN_TEMPLATE(UINT32, uint32_t);
            DECLARE_COMPUTE_RUN_TEMPLATE(INT64, int64_t);
            DECLARE_COMPUTE_RUN_TEMPLATE(UINT64, uint64_t);
            DECLARE_COMPUTE_RUN_TEMPLATE(FLOAT32, float);
            DECLARE_COMPUTE_RUN_TEMPLATE(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN_TEMPLATE
            default: {
                TS_LOG_ERROR << "_cast not support this data type: " << to_type << eject;
                break;
            }
        }
    }



    void CastV2::cast(const Tensor &x, DTYPE to_type, Tensor &out) {
        // Notice: the all tensor' memory device are CPU, as given in running_memory_device
        DTYPE dtype = x.dtype();
        switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_cast_compute_run<TYPE>(x, to_type, out); break; }
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
                TS_LOG_ERROR << this->op() << " not support this data type: " << dtype << eject;
                break;
            }
        }
    }

  }
}
/////////////////////////////////////////////////

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(CastV2, GPU, name::layer::cast())

