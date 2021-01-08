#include <kernels/gpu/equal.h>
#include "backend/name.h"
#include "global/operator_factory.h"
#include "global/fp16_operator_factory.h"

#include <cuda_fp16.h>
#include "kernels/gpu/cudax_fp16_math.h"
#include <kernels/gpu/gpu_kernel.h>


namespace ts {
    namespace gpu {

        template<typename T>
        static __global__ void reduce_operator_kernel(uint8_t *out, int size, const T* lhs, const T* rhs,
                                               int *lhsshape, int *lhsweight,
                                               int *rhsshape, int *rhsweight,
                                               int *outweight, int shapelen) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size)
                return;

            int *ptmp = outweight + 1;
            int ntmp = index;

            int rhsindex = 0;
            int lhsindex = 0;
            int nbuff1,nbuff2;
            nbuff1 = nbuff2 = 0;
            for(int m = 0, i= shapelen - 1; i >= 0; --i, m++) {
                if(i > 0) {
                    nbuff1 = ntmp / *ptmp;
                    ntmp %= *ptmp;
                }else {
                    nbuff1 = ntmp;
                }

                nbuff2 = nbuff1 % lhsshape[m];
                if(m < shapelen - 1) {
                    lhsindex += nbuff2 * lhsweight[m+1];
                }else {
                    lhsindex += nbuff2;
                }

                nbuff2 = nbuff1 % rhsshape[m];

                if(m < shapelen - 1) {
                    rhsindex += nbuff2 * rhsweight[m+1];
                }else {
                    rhsindex += nbuff2;
                }

                ++ptmp;
            }
            out[index] = rhs[rhsindex] == lhs[lhsindex];
        }

        template<typename T>
        static __global__ void reduce_operator_scalar(uint8_t *out, int size, const T* lhs, const T* rhs) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size) {
                out[index] = lhs[index] == rhs[0];
            }
        }

        template<typename T>
        static __global__ void reduce_operator_same_shape_kernel(uint8_t *out, int size, const T* lhs, const T* rhs) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size) {
                out[index] = lhs[index] == rhs[index];
            }
        }

#ifdef TS_USE_CUDA_FP16
        template<>
        __global__ void reduce_operator_kernel(uint8_t *out, int size, const half* lhs, const half* rhs,
                                                      int *lhsshape, int *lhsweight,
                                                      int *rhsshape, int *rhsweight,
                                                      int *outweight, int shapelen) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size)
                return;

            int *ptmp = outweight + 1;
            int ntmp = index;

            int rhsindex = 0;
            int lhsindex = 0;
            int nbuff1,nbuff2;
            nbuff1 = nbuff2 = 0;
            for(int m = 0, i= shapelen - 1; i >= 0; --i, m++) {
                if(i > 0) {
                    nbuff1 = ntmp / *ptmp;
                    ntmp %= *ptmp;
                }else {
                    nbuff1 = ntmp;
                }

                nbuff2 = nbuff1 % lhsshape[m];
                if(m < shapelen - 1) {
                    lhsindex += nbuff2 * lhsweight[m+1];
                }else {
                    lhsindex += nbuff2;
                }

                nbuff2 = nbuff1 % rhsshape[m];

                if(m < shapelen - 1) {
                    rhsindex += nbuff2 * rhsweight[m+1];
                }else {
                    rhsindex += nbuff2;
                }

                ++ptmp;
            }
            out[index] = rhs[rhsindex] == lhs[lhsindex];
        }

        template<>
        __global__ void reduce_operator_scalar(uint8_t *out, int size, const half* lhs, const half* rhs) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size) {
                out[index] = lhs[index] == rhs[0];
            }
        }

        template<>
        __global__ void reduce_operator_same_shape_kernel(uint8_t *out, int size, const half* lhs, const half* rhs) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size) {
                out[index] = lhs[index] == rhs[index];
            }
        }
#endif


        template<typename T>
        static inline void equal_gpu_compute_run_broadcast(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            HypeShape lhs_hype(lhs.sizes());
            HypeShape rhs_hype(rhs.sizes());
            HypeShape out_hype(out.sizes());

            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<uint8_t>();

            auto ncount = out.count();

            int *lhsshape = nullptr;
            int *rhsshape = nullptr;
            int *lhsweight = nullptr;
            int *rhsweight = nullptr;
            int *outweight = nullptr;

            Shape tmpshape;
            tmpshape.resize(1);
            tmpshape[0] = int32_t(lhs.sizes().size());
            Tensor lhs_tensor(out.device(), INT32, tmpshape);
            lhsshape = lhs_tensor.data<int32_t>();

            tmpshape[0] = int32_t(rhs.sizes().size());
            Tensor rhs_tensor(out.device(), INT32, tmpshape);
            rhsshape = rhs_tensor.data<int32_t>();

            tmpshape[0] = int32_t(lhs.sizes().size());
            Tensor lhs_weight_tensor(out.device(), INT32, tmpshape);
            lhsweight = lhs_weight_tensor.data<int32_t>();

            tmpshape[0] = int32_t(rhs.sizes().size());
            Tensor rhs_weight_tensor(out.device(), INT32, tmpshape);
            rhsweight = rhs_weight_tensor.data<int32_t>();

            tmpshape[0] = int32_t(out.sizes().size());
            Tensor out_weight_tensor(out.device(), INT32, tmpshape);
            outweight = out_weight_tensor.data<int32_t>();

            memcpy((void*)lhsshape, out.device(), lhs.sizes().size() * sizeof(int32_t),
                   (void*)lhs.sizes().data(), MemoryDevice(CPU), lhs.sizes().size() * sizeof(int32_t));

            memcpy((void*)rhsshape, out.device(), rhs.sizes().size() * sizeof(int32_t),
                   (void*)rhs.sizes().data(), MemoryDevice(CPU), rhs.sizes().size() * sizeof(int32_t));

            memcpy((void*)lhsweight, out.device(), lhs_hype.weight().size() * sizeof(int32_t),
                   (void*)lhs_hype.weight().data(), MemoryDevice(CPU), lhs_hype.weight().size() * sizeof(int32_t));

            memcpy((void*)rhsweight, out.device(), rhs_hype.weight().size() * sizeof(int32_t),
                   (void*)rhs_hype.weight().data(), MemoryDevice(CPU), rhs_hype.weight().size() * sizeof(int32_t));

            memcpy((void*)outweight, out.device(), out_hype.weight().size() * sizeof(int32_t),
                   (void*)out_hype.weight().data(), MemoryDevice(CPU), out_hype.weight().size() * sizeof(int32_t));

            RUN_KERNEL(reduce_operator_kernel<T>, CUDA_BLOCK(ncount, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       pout, ncount, plhs, prhs, lhsshape, lhsweight, rhsshape, rhsweight, outweight, int(out.sizes().size()));

        }



        template<typename T>
        static inline void equal_gpu_compute_run_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<uint8_t>();

            RUN_KERNEL(reduce_operator_scalar<T>, CUDA_BLOCK(out.count(), CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       pout, out.count(), plhs, prhs);
        }


        template<typename T>
        static inline void equal_gpu_compute_run_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<uint8_t>();

            RUN_KERNEL(reduce_operator_same_shape_kernel<T>, CUDA_BLOCK(out.count(), CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       pout, out.count(), plhs, prhs);
        }

        void Equal::reduce_with_broadcast(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            DTYPE dtype = lhs.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { equal_gpu_compute_run_broadcast<TYPE>(lhs, rhs, out); break; }
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
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype)
                                 << eject;
                    break;
                }
            }
        }

        void Equal::reduce_with_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            DTYPE dtype = lhs.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { equal_gpu_compute_run_scalar<TYPE>(lhs, rhs, out); break; }
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
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype)
                                 << eject;
                    break;
                }
            }
        }

        void Equal::reduce_with_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            DTYPE dtype = lhs.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { equal_gpu_compute_run_same_shape<TYPE>(lhs, rhs, out); break; }
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
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype)
                                 << eject;
                    break;
                }
            }
        }

        void Equal::reduce_with_scalar_cross(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            this->reduce_with_scalar(rhs, lhs, out);
        }
    }
}

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(Equal, GPU, "equal")
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(Equal, GPU, "equal")
#endif
