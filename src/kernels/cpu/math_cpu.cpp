//
// Created by seeta on 2018/7/19.
//

#include "kernels/cpu/math_cpu.h"

#include <iostream>
#include <cassert>
#include <cmath>

namespace ts {
    namespace cpu {
        template<typename T>
        static inline T inline_dot(int N, const T *x, int incx, const T *y, int incy) {
            T sum = 0;
            // block: 4
            int i = 0;
            static const int block_size = 4;
            int blocked_N = N % block_size ? N - block_size : N;
            for (; i < blocked_N; i += block_size) {
                sum += *x * *y; x += incx; y += incy;
                sum += *x * *y; x += incx; y += incy;
                sum += *x * *y; x += incx; y += incy;
                sum += *x * *y; x += incx; y += incy;
            }
            for (; i < N; ++i) {
                sum += *x * *y; x += incx; y += incy;
            }
            return sum;
        }

        template<typename T>
        static inline void inline_zero(int N, T *x, int incx) {
            // block: 4
            int i = 0;
            static const int block_size = 4;
            int blocked_N = N % block_size ? N - block_size : N;
            for (; i < blocked_N; i += block_size) {
                *x = 0; x += incx;
                *x = 0; x += incx;
                *x = 0; x += incx;
                *x = 0; x += incx;
            }
            for (; i < N; ++i) {
                *x = 0; x += incx;
            }
        }

        template<typename T>
        static inline void inline_scal(int N, T alpha, T *x, int incx) {
            if (alpha == 1) return; // TODO: update float number equal check method
            if (alpha == 0) {
                inline_zero<T>(N, x, incx);
                return;
            }
            // block: 4
            int i = 0;
            static const int block_size = 4;
            int blocked_N = N % block_size ? N - block_size : N;
            for (; i < blocked_N; i += block_size) {
                *x *= alpha; x += incx;
                *x *= alpha; x += incx;
                *x *= alpha; x += incx;
                *x *= alpha; x += incx;
            }
            for (; i < N; ++i) {
                *x *= alpha; x += incx;
            }
        }


        template<typename T>
        T math<T>::dot(int N, const T *x, int incx, const T *y, int incy) {
            return inline_dot<T>(N, x, incx, y, incy);
        }

        template<typename T>
        static inline void inline_gemm_row_major(
                blas::Transpose TransA,
                blas::Transpose TransB,
                int M, int N, int K,
                T alpha,
                const T *A, int lda,
                const T *B, int ldb,
                T beta,
                T *C, int ldc) {
            // TODO: check if lda, ldb, ldc use correct
            assert(lda >= (TransA == blas::NoTrans ? K : M));
            assert(ldb >= (TransB == blas::NoTrans ? N : K));
            assert(ldc >= N);

            // calculate beta * C
            // C is RowMajor
            if (ldc == N) inline_scal(M * N, beta, C, 1);
            else {
                T *C_anchor = C;
                for (int i = 0; i < M; ++i, C += ldc) inline_scal(N, beta, C_anchor, 1);
            }

            if (alpha == 0) return;

            unsigned int condition = (TransA == blas::NoTrans ? 0U : 1U) | ((TransB == blas::NoTrans ? 0U : 2U));
            switch (condition) {
                case 0: // A: NoTrans, B: NoTrans
                    for (int i = 0; i < M; ++i) {
                        T *C_anchor = &C[i * ldc];
                        for (int j = 0; j < N; ++j) {
                            *C_anchor += alpha * inline_dot(K, &A[i * lda], 1, &B[j], ldb);
                            C_anchor++;
                        }
                    }
                    break;
                case 1: // A: Trans, B: NoTrans
                    for (int i = 0; i < M; ++i) {
                        T *C_anchor = &C[i * ldc];
                        for (int j = 0; j < N; ++j) {
                            *C_anchor += alpha * inline_dot(K, &A[i], lda, &B[j], ldb);
                            C_anchor++;
                        }
                    }
                    break;
                case 2: // A: NoTrans, B: Trans
                    for (int i = 0; i < M; ++i) {
                        T *C_anchor = &C[i * ldc];
                        for (int j = 0; j < N; ++j) {
                            *C_anchor += alpha * inline_dot(K, &A[i * lda], 1, &B[j * ldb], 1);
                            C_anchor++;
                        }
                    }
                    break;
                default: // A: Trans, B: Trans
                    for (int i = 0; i < M; ++i) {
                        T *C_anchor = &C[i * ldc];
                        for (int j = 0; j < N; ++j) {
                            *C_anchor += alpha * inline_dot(K, &A[i], lda, &B[j * ldb], 1);
                            C_anchor++;
                        }
                    }
                    break;
            }
        }

        template<typename T>
        void
        math<T>::gemm(
                blas::Order Order,
                blas::Transpose TransA,
                blas::Transpose TransB,
                int M, int N, int K,
                T alpha,
                const T *A, int lda,
                const T *B, int ldb,
                T beta,
                T *C, int ldc) {
            if (Order == blas::ColMajor) {
                inline_gemm_row_major<T>(TransB, TransA, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc);
            } else {
                inline_gemm_row_major<T>(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            }
        }

        template<typename T>
        T math<T>::dot(int N, const T *x, const T *y) {
            return dot(N, x, 1, y, 1);
        }

        template<typename T>
        void math<T>::gemm(blas::Transpose TransA, blas::Transpose TransB, int M, int N, int K, T alpha, const T *A,
                           const T *B, T beta, T *C) {
            int lda = (TransA == blas::NoTrans ? K : M);
            int ldb = (TransB == blas::NoTrans ? N : K);
            int ldc = N;
            inline_gemm_row_major<T>(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }

        template<typename T>
        T math<T>::asum(int N, const T *x, int incx) {
            T sum = 0;
            // block: 4
            int i = 0;
            static const int block_size = 4;
            int blocked_N = N % block_size ? N - block_size : N;
            for (; i < blocked_N; i += block_size) {
                sum += abs(*x); x += incx;
                sum += abs(*x); x += incx;
                sum += abs(*x); x += incx;
                sum += abs(*x); x += incx;
            }
            for (; i < N; ++i) {
                sum += abs(*x); x += incx;
            }
            return sum;
        }

        template<typename T>
        T math<T>::abs(T val) {
            return std::fabs(val);
        }
    }
}

template class ts::cpu::math<ts::type<ts::FLOAT32>::declare>;
template class ts::cpu::math<ts::type<ts::FLOAT64>::declare>;