//
// Created by kier on 2018/7/19.
//

#ifndef TENSORSTACK_KERNELS_CPU_MATH_CPU_H
#define TENSORSTACK_KERNELS_CPU_MATH_CPU_H

#include "core/tensor.h"
#include "../common/blas.h"

namespace ts {
    namespace cpu {
        template <typename T>
        class TS_DEBUG_API math {
        public:
            static void check(const Tensor &tensor) {
                if (tensor.device().type() != CPU) throw DeviceMismatchException(Device(CPU), tensor.device());
            }

            static T abs(T val);

            static T dot(
                    int N,
                    const T *x,
                    int incx,
                    const T *y,
                    int incy
                    );

            static T dot(int N, const T *x, const T *y);

            static void gemm(
                    blas::Order Order,
                    blas::Transpose TransA,
                    blas::Transpose TransB,
                    int M, int N, int K,
                    T alpha,
                    const T *A, int lda,
                    const T *B, int ldb,
                    T beta,
                    T *C, int ldc);

            static void gemm(
                    blas::Transpose TransA,
                    blas::Transpose TransB,
                    int M, int N, int K,
                    T alpha, const T *A, const T *B,
                    T beta, T *C);

            static T asum(
                    int N,
                    const T *x,
                    int incx
                    );
        };
    }
}

extern template class ts::cpu::math<ts::dtype<ts::FLOAT32>::declare>;
extern template class ts::cpu::math<ts::dtype<ts::FLOAT64>::declare>;


#endif //TENSORSTACK_KERNELS_CPU_MATH_CPU_H
