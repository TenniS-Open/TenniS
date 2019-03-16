#include "kernels/gpu/math_cublas.h"
#include "utils/platform.h"

#include "kernels/gpu/math_gpu.h"

namespace ts {
    namespace gpu {
        namespace cublas{

            template<typename T>
            inline bool cublas_dot(cublasHandle_t handle, int N, const T *x,int incx, const T *y, int incy, T *z ) {
                return gpu::math<T>::dot(N, x, incx, y, incy, z);
            }

            template<>
            inline bool cublas_dot<float>(cublasHandle_t handle,int N, const float *x,int incx, const float *y, int incy, float *z) {
                if (CUBLAS_STATUS_SUCCESS == cublasSdot(handle, N, x, incx, y, incy, z))
                    return true;
                return false;
            }

            template<>
            inline bool cublas_dot<double>(cublasHandle_t handle,int N, const double *x,int incx, const double *y, int incy, double *z) {
                if (CUBLAS_STATUS_SUCCESS == cublasDdot(handle, N, x, incx, y, incy, z))
                    return true;
                return false;
            }

            template<typename T>
            bool math<T>::dot(cublasHandle_t handle,int N, const T *x, int incx, const T *y, int incy, T *z) {
                return cublas_dot<T>(handle,N, x, incx, y, incy,z);
            }

            template<typename T>
            bool math<T>::dot(cublasHandle_t handle, int N, const T *x, const T *y, T *z) {
                return dot(handle, N, x, 1, y, 1,z);
            }

            template<typename T>
            bool cblas_gemm(cublasHandle_t handle, Order Order, Transpose TransA, Transpose TransB,
                int M, int N, int K,
                T alpha, const T *A, int lda,
                const T *B, int ldb, T beta,
                T *C, int ldc) {
                return gpu::math<T>::gemm(Order,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
            }

            template<>
            bool cblas_gemm<float>(cublasHandle_t handle,Order Order, Transpose TransA, Transpose TransB,
                int M, int N, int K,
                float alpha, const float *A, int lda,
                const float *B, int ldb, float beta,
                float *C, int ldc) {
                cublasOperation_t cblas_TransA = TransA == ts::gpu::cublas::NoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
                cublasOperation_t cblas_TransB = TransB == ts::gpu::cublas::NoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
                if (Order == ColMajor) {
                    if (CUBLAS_STATUS_SUCCESS == cublasSgemm(handle, cblas_TransA, cblas_TransB,
                        M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc))
                        return true;
                    return false;
                }
                else {
                    if (CUBLAS_STATUS_SUCCESS == cublasSgemm(handle, cblas_TransB, cblas_TransA,
                        N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc))
                        return true;
                    return false;
                }
            }

            template<>
            bool cblas_gemm<double>(cublasHandle_t handle,Order Order, Transpose TransA, Transpose TransB,
                int M, int N, int K,
                double alpha, const double *A, int lda,
                const double *B, int ldb, double beta,
                double *C, int ldc) {
                cublasOperation_t cblas_TransA = TransA == ts::gpu::cublas::NoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
                cublasOperation_t cblas_TransB = TransB == ts::gpu::cublas::NoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
                if (Order == ColMajor){
                    if (CUBLAS_STATUS_SUCCESS == cublasDgemm(handle, cblas_TransA, cblas_TransB,
                        M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc))
                        return true;
                    return false;
                }
                else {
                    if (CUBLAS_STATUS_SUCCESS == cublasDgemm(handle, cblas_TransA, cblas_TransB,
                        N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc))
                        return true;
                    return false;
                }
            }

            template<typename T>
            bool math<T>::gemm(cublasHandle_t handle,Order Order, Transpose TransA, Transpose TransB,
                    int M, int N, int K,
                    T alpha, const T *A, int lda,
                    const T *B, int ldb, T beta,
                    T *C, int ldc) {
                return cblas_gemm(handle,Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            }

            template<typename T>
            bool math<T>::gemm(cublasHandle_t handle, Transpose TransA, Transpose TransB,
                int M, int N, int K,
                T alpha, const T *A, const T *B, 
                T beta,T *C) {
                cublas::Order Order = cublas::RowMajor;
                int lda = (TransA == cublas::NoTrans ? K : M);
                int ldb = (TransB == cublas::NoTrans ? N : K);
                int ldc = N;

                return cblas_gemm(handle, Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            }

            template <typename T>
            bool cublas_asum(cublasHandle_t handle, int N, const T *x, int incx,T* out) {
                return gpu::math<T>::asum(N, x, incx,out);
            }

            template <>
            bool cublas_asum<float>(cublasHandle_t handle, int N, const float *x, int incx,float *out) {
                if (cudaSuccess == cublasSasum(handle, N, x, incx, out))
                    return true;
                return false;
            }

            template <>
            bool cublas_asum<double>(cublasHandle_t handle, int N, const double *x, int incx, double *out) {
                if (cudaSuccess == cublasDasum(handle, N, x, incx, out))
                    return true;
                return false;
            }

            template<typename T>
            bool math<T>::asum(cublasHandle_t handle, int N, const T *x, int incx, T *out) {
                return cublas_asum(handle,N, x, incx,out);
            }
        }

    }
}

template class ts::gpu::cublas::math<ts::dtype<ts::FLOAT32>::declare>;
template class ts::gpu::cublas::math<ts::dtype<ts::FLOAT64>::declare>;
