#include <kernels/gpu/batch_norm.h>
#include <core/tensor_builder.h>

#include <global/operator_factory.h>
#include <backend/name.h>
#include <utils/assert.h>
#include <core/device.h>
#include <vector>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <math_functions.hpp>


namespace ts {
    namespace gpu {
        template<typename T>
        static __global__ void gpu_batch_norm_compute_kernel(T* data, int size, int step, int slice,
                                        const T* mean, const T* variance ) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size) {
                int dim = index % ( step * slice ) / (step);
                data[index] = (data[index] - mean[dim]) / variance[dim];
            }
        }


        template<typename T>
        static void gpu_batch_norm_compute_run(const Tensor &x, const Tensor &mean,
                                               const Tensor &variance, int dim, float epsilon, Tensor &out) {
            const Shape &shape = x.sizes();
            //int predims = 1;
            int backdims = 1;
            //for (int i = 0; i < dim; i++) {
            //    predims *= shape[i];
            //}

            for (int i = dim + 1; i < shape.size(); i++) {
                backdims *= shape[i];
            }

            const T *psrc = x.data<T>();
            const T *pmean = mean.data<T>();
            const T *pvariance = variance.data<T>();
            T *pdst = out.data<T>();

            std::vector<T> vec(variance.count());

            memcpy((void*)vec.data(), MemoryDevice(CPU), vec.size() * sizeof(T),
                   (void*)pvariance, variance.device(), vec.size() * sizeof(T));


            for (int i = 0; i < vec.size(); i++) {
                vec[i] = T(1) / sqrt(vec[i] + T(epsilon));
            }

            T * pvar = nullptr;

            Shape tmpshape;
            tmpshape.resize(1);
            tmpshape[0] = vec.size();
            Tensor variance_tensor(variance.device(), variance.dtype(), tmpshape);
            pvar = variance_tensor.data<T>();

            memcpy((void*)pvar, variance.device(), vec.size() * sizeof(T),
                   (void*)vec.data(), MemoryDevice(CPU), vec.size() * sizeof(T));


            memcpy((void*)pdst, out.device(), out.count() * sizeof(T),
                   (void*)psrc, x.device(), x.count() * sizeof(T));

            gpu_batch_norm_compute_kernel<T> <<< CUDA_BLOCK(out.count(), CUDA_THREAD_NUM), CUDA_THREAD_NUM >>> (pdst, out.count(), backdims, shape[dim], pmean, pvar);
           
        }


        void BatchNorm::batch_norm(const Tensor &x, const Tensor &mean, const Tensor &variance,
                                   int dim, float epsilon, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_batch_norm_compute_run<TYPE>(x, mean, variance, dim, epsilon, out); break; }
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

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(BatchNorm, GPU, name::layer::batch_norm())
