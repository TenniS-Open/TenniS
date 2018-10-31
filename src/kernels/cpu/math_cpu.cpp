//
// Created by seeta on 2018/7/19.
//

#include "kernels/cpu/math_cpu.h"
#include "kernels/common/math.h"

#include <iostream>
#include <cassert>
#include <cmath>

#ifdef TS_USE_SSE
#include <immintrin.h>
#endif

namespace ts {
    namespace cpu {
        template<typename T>
        inline T inline_dot(int N, const T *x, int incx, const T *y, int incy) {
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
#ifdef TS_USE_SSE

        inline float inline_dot_conitnous_float(int N, const float *x, const float *y) {
            float sum = 0;
            // block: 4
            int i = 0;
            static const int block_size = 4;
            int blocked_N = N % block_size ? N - block_size : N;
            __m128 simdX, simdY;
            __m128 simdSUM = _mm_setzero_ps();
            float simdBuffer[4];
            for (; i < blocked_N; i += block_size) {
                simdX = _mm_loadu_ps(x);
                simdY = _mm_loadu_ps(y);
                x += 4;
                y += 4;
                simdSUM = _mm_add_ps(simdSUM, _mm_mul_ps(simdX, simdY));
            }
            _mm_storeu_ps(simdBuffer, simdSUM);
            sum = simdBuffer[0] + simdBuffer[1] + simdBuffer[2] + simdBuffer[3];
            for (; i < N; ++i) {
                sum += *x * *y; x += 1; y += 1;
            }
            return sum;
        }
        template <>
        inline float inline_dot<float>(int N, const float *x, int incx, const float *y, int incy) {
            if (incx == 1 && incy == 1) return inline_dot_conitnous_float(N, x, y);
            float sum = 0;
            // block: 4
            int i = 0;
            static const int block_size = 4;
            int blocked_N = N % block_size ? N - block_size : N;
            __m128 simdX, simdY;
            __m128 simdSUM = _mm_setzero_ps();
            float simdBuffer[4];
            for (; i < blocked_N; i += block_size) {
                simdBuffer[0] = *x; x += incx;
                simdBuffer[1] = *x; x += incx;
                simdBuffer[2] = *x; x += incx;
                simdBuffer[3] = *x; x += incx;
                simdX = _mm_loadu_ps(simdBuffer);
                simdBuffer[0] = *y; y += incy;
                simdBuffer[1] = *y; y += incy;
                simdBuffer[2] = *y; y += incy;
                simdBuffer[3] = *y; y += incy;
                simdY = _mm_loadu_ps(simdBuffer);
                simdSUM = _mm_add_ps(simdSUM, _mm_mul_ps(simdX, simdY));
            }
            _mm_storeu_ps(simdBuffer, simdSUM);
            sum = simdBuffer[0] + simdBuffer[1] + simdBuffer[2] + simdBuffer[3];
            for (; i < N; ++i) {
                sum += *x * *y; x += incx; y += incy;
            }
            return sum;
        }
#endif

        template<typename T>
        inline void inline_zero(int N, T *x, int incx) {
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
        inline void inline_scal(int N, T alpha, T *x, int incx) {
            if (ts::near(alpha, 1)) return; // TODO: update float number equal check method
            if (ts::near(alpha, 0)) {
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
        inline void inline_gemm_row_major(
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

            if (ts::near(alpha, 0)) return;

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

template class ts::cpu::math<ts::dtype<ts::FLOAT32>::declare>;
template class ts::cpu::math<ts::dtype<ts::FLOAT64>::declare>;