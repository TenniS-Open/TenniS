#include "kernels/gpu/where.h"
#include "backend/name.h"
#include "global/operator_factory.h"

#include "global/fp16_operator_factory.h"

#include "kernels/gpu/cudax_fp16_math.h"
#include <kernels/gpu/gpu_kernel.h>


namespace ts {
    namespace gpu {

        template<typename T>
        static __global__ void same_shape_kernels(T *out, int size, const uint8_t *cond, const T *lhs, const T *rhs) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size) {
                out[index] = cond[index] ? lhs[index] : rhs[index];
            }
        }

        template<typename T>
        static __global__ void broadcast_kernels(T *out, int size, const uint8_t *cond, const T *lhs, const T *rhs,
                                                 int *condshape, int *lhsshape, int *rhsshape,
                                                 int *condweight, int *lhsweight, int *rhsweight,
                                                 int *outweight, int shapelen) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size)
                return;

            int *ptmp = outweight + 1;
            int ntmp = index;

            int condindex = 0;
            int rhsindex = 0;
            int lhsindex = 0;
            int nbuff1, nbuff2;
            nbuff1 = nbuff2 = 0;
            for (int m = 0, i = shapelen - 1; i >= 0; --i, m++) {
                if (i > 0) {
                    nbuff1 = ntmp / *ptmp;
                    ntmp %= *ptmp;
                } else {
                    nbuff1 = ntmp;
                }

                nbuff2 = nbuff1 % condshape[m];
                if (m < shapelen - 1) {
                    condindex += nbuff2 * condweight[m + 1];
                } else {
                    condindex += nbuff2;
                }

                nbuff2 = nbuff1 % lhsshape[m];
                if (m < shapelen - 1) {
                    lhsindex += nbuff2 * lhsweight[m + 1];
                } else {
                    lhsindex += nbuff2;
                }

                nbuff2 = nbuff1 % rhsshape[m];

                if (m < shapelen - 1) {
                    rhsindex += nbuff2 * rhsweight[m + 1];
                } else {
                    rhsindex += nbuff2;
                }

                ++ptmp;
            }

            out[index] = cond[condindex] ? lhs[lhsindex] : rhs[rhsindex];
        }




        template<typename T>
        static inline void
        gpu_compute_run_broadcast(const Tensor &cond, const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            HypeShape cond_hype(cond.sizes());
            HypeShape lhs_hype(lhs.sizes());
            HypeShape rhs_hype(rhs.sizes());
            HypeShape out_hype(out.sizes());

            auto pcond = cond.data<uint8_t>();
            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<T>();

            auto ncount = out.count();

            int *condshape = nullptr;
            int *lhsshape = nullptr;
            int *rhsshape = nullptr;
            int *condweight = nullptr;
            int *lhsweight = nullptr;
            int *rhsweight = nullptr;
            int *outweight = nullptr;

            /////////////////////////////////////
            Shape tmpshape;
            tmpshape.resize(1);
            tmpshape[0] = int32_t(cond.sizes().size());
            Tensor cond_tensor(out.device(), INT32, tmpshape);
            condshape = cond_tensor.data<int32_t>();

            tmpshape.resize(1);
            tmpshape[0] = int32_t(lhs.sizes().size());
            Tensor lhs_tensor(out.device(), INT32, tmpshape);
            lhsshape = lhs_tensor.data<int32_t>();

            tmpshape[0] = int32_t(rhs.sizes().size());
            Tensor rhs_tensor(out.device(), INT32, tmpshape);
            rhsshape = rhs_tensor.data<int32_t>();

            tmpshape[0] = int32_t(lhs.sizes().size());
            Tensor cond_weight_tensor(out.device(), INT32, tmpshape);
            condweight = cond_weight_tensor.data<int32_t>();

            tmpshape[0] = int32_t(lhs.sizes().size());
            Tensor lhs_weight_tensor(out.device(), INT32, tmpshape);
            lhsweight = lhs_weight_tensor.data<int32_t>();

            tmpshape[0] = int32_t(rhs.sizes().size());
            Tensor rhs_weight_tensor(out.device(), INT32, tmpshape);
            rhsweight = rhs_weight_tensor.data<int32_t>();

            tmpshape[0] = int32_t(out.sizes().size());
            Tensor out_weight_tensor(out.device(), INT32, tmpshape);
            outweight = out_weight_tensor.data<int32_t>();

            memcpy((void *) condshape, out.device(), cond.sizes().size() * sizeof(int32_t),
                   (void *) cond.sizes().data(), MemoryDevice(CPU), cond.sizes().size() * sizeof(int32_t));

            memcpy((void *) lhsshape, out.device(), lhs.sizes().size() * sizeof(int32_t),
                   (void *) lhs.sizes().data(), MemoryDevice(CPU), lhs.sizes().size() * sizeof(int32_t));

            memcpy((void *) rhsshape, out.device(), rhs.sizes().size() * sizeof(int32_t),
                   (void *) rhs.sizes().data(), MemoryDevice(CPU), rhs.sizes().size() * sizeof(int32_t));

            memcpy((void *) condweight, out.device(), cond_hype.weight().size() * sizeof(int32_t),
                   (void *) cond_hype.weight().data(), MemoryDevice(CPU), cond_hype.weight().size() * sizeof(int32_t));

            memcpy((void *) lhsweight, out.device(), lhs_hype.weight().size() * sizeof(int32_t),
                   (void *) lhs_hype.weight().data(), MemoryDevice(CPU), lhs_hype.weight().size() * sizeof(int32_t));

            memcpy((void *) rhsweight, out.device(), rhs_hype.weight().size() * sizeof(int32_t),
                   (void *) rhs_hype.weight().data(), MemoryDevice(CPU), rhs_hype.weight().size() * sizeof(int32_t));

            memcpy((void *) outweight, out.device(), out_hype.weight().size() * sizeof(int32_t),
                   (void *) out_hype.weight().data(), MemoryDevice(CPU), out_hype.weight().size() * sizeof(int32_t));

            RUN_KERNEL(broadcast_kernels<T>, CUDA_BLOCK(ncount, CUDA_THREAD_NUM), CUDA_THREAD_NUM, pout, ncount,
                       pcond, plhs, prhs, condshape, lhsshape, rhsshape, condweight, lhsweight, rhsweight, outweight,
                       int(out.sizes().size()));
        }


        template<typename T>
        static inline void
        gpu_compute_run_same_shape(const Tensor &cond, const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            auto pcond = cond.data<uint8_t>();
            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<T>();

            auto ncount = out.count();

            RUN_KERNEL(same_shape_kernels<T>, CUDA_BLOCK(out.count(), CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       pout, ncount, pcond, plhs, prhs);

        }

        void Where::reduce_with_broadcast(const Tensor &cond, const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_compute_run_broadcast<TYPE>(cond, lhs, rhs, out); break; }
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

        void Where::reduce_with_same_shape(const Tensor &cond, const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_compute_run_same_shape<TYPE>(cond, lhs, rhs, out); break; }
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
    }
}

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(Where, GPU, "where")
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(Where, GPU, "where")
#endif
