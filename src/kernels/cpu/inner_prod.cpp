#include <kernels/cpu/inner_prod.h>
#include <core/tensor_builder.h>
#include <kernels/cpu/math_cpu.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

#ifdef TS_USE_CBLAS
#include <kernels/cblas/math_cblas.h>
#endif



namespace ts {
    namespace cpu {
        template<typename T>
        static void cpu_inner_prod_compute_run(const Tensor &lhs, const Tensor &rhs, bool transpose, Tensor &out) {
            const Shape &lhs_shape = lhs.sizes();
            const Shape &rhs_shape = rhs.sizes();
            // const Shape &out_shape = out.sizes();

            const T *psrc = lhs.data<T>();
            const T *pdot = rhs.data<T>();
            T *pdst = out.data<T>();

            auto rhs_transpose = transpose ? blas::Trans : blas::NoTrans;
            auto N = transpose ? rhs_shape[0] : rhs_shape[1];
#ifdef TS_USE_CBLAS
            cblas::math<T>::gemm(blas::NoTrans, rhs_transpose, lhs_shape[0], N, lhs_shape[1],
                                 (T) 1, psrc, pdot, (T) 0, pdst);
#else
            cpu::math<T>::gemm(blas::NoTrans, rhs_transpose, lhs_shape[0], N, lhs_shape[1],
                               (T) 1, psrc, pdot, (T) 0, pdst);
#endif
        }

        void InnerProd::inner_prod(const Tensor &lhs, const Tensor &rhs, bool transpose, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_inner_prod_compute_run<TYPE>(lhs, rhs, transpose, out); break; }
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
using namespace cpu;
TS_REGISTER_OPERATOR(InnerProd, CPU, name::layer::inner_prod())
