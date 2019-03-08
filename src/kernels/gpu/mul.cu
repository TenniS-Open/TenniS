#include <kernels/gpu/mul.h>
#include <core/tensor_builder.h>
#include <backend/name.h>
#include <utils/assert.h>
#include <global/operator_factory.h>
#include <core/device.h>

#include <numeric>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>


//#ifdef TS_USE_OPENMP
//#include "kernels/common/openmp.h"
//#endif

namespace ts {
    namespace gpu {

        template<typename T>
        static __global__ void reduce_operator_scalar_kernel(T* data, int size, const T *scalar) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size) {
                data[index] *= *scalar;
            }
        }

        template<typename T>
        static __global__ void reduce_operator_same_shape_kernel(T* data, const T*bias, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size) {
                //int dim = index % ( step * slice ) / (step);
                data[index] *= bias[index];
            }
        }

        template<typename T>
        static __global__ void reduce_operator_bias_kernel(T* data, int size, int step, int slice,
                                        const T* bias, int biaslen ) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size) {
                int dim = index % ( step * slice ) / (step);
                data[index] *= bias[dim];
            }
        }


        template<typename T>
        static __global__ void reduce_operator_kernel(T* out, int size, const T* lhs,  const T* rhs,
                                               int *lhsshape, int *lhsweight,
                                               int *rhsshape, int *rhsweight,
                                               int *outweight, int shapelen) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size)
                return;

            int *buffer1 = new int[shapelen];
            int *buffer2 = new int[shapelen];
            int i = 0;
            int k= 0;
            int *  ptmp;
            int *  ptr;

            ptr = buffer1;
            ptmp = outweight + 1;
            int ntmp = index;
            for(i= shapelen - 1; i; --i) {
                *ptr = ntmp / *ptmp;
                ntmp %= *ptmp;
                ++ptmp;
                ++ptr;
            }

            *ptr = ntmp;

            int lhsindex = 0;
            for(i=0; i<shapelen; ++i) {
                buffer2[i] = buffer1[i] % lhsshape[i];
            }

            for(k=0, i=1; i < shapelen; ++k,++i) {
                lhsindex += buffer2[k] * lhsweight[i];
            }
            lhsindex += buffer2[k];

            int rhsindex = 0;
            for(i=0; i<shapelen; ++i) {
                buffer2[i] = buffer1[i] % rhsshape[i];
            }

            for(k=0, i=1; i < shapelen; ++k,++i) {
                rhsindex += buffer2[k] * rhsweight[i];
            }
            rhsindex += buffer2[k];

            out[index] = lhs[lhsindex] * rhs[rhsindex];

            delete [] buffer1;
            delete [] buffer2;
        }



        /* 
        template <typename T>
        inline void reduce_operator(T &x, T lhs, T rhs) {
            x = lhs * rhs;
        }
        template <typename T>
        inline void reduce_operator(T &x, T y) {
            x *= y;
        }

        static inline int to_mod_index(const HypeShape &hype, const std::vector<int> &coordinate) {
            auto temp = coordinate;
            for (size_t i = 0; i < temp.size(); ++i) {
                temp[i] %= hype.shape(i);
            }
            return hype.to_index(temp);
        }
        */

        template<typename T>
        static inline void mul_gpu_compute_run(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            HypeShape lhs_hype(lhs.sizes());
            HypeShape rhs_hype(rhs.sizes());
            HypeShape out_hype(out.sizes());
            //ShapeIterator out_iterator(out.sizes());

            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<T>();

            auto ncount = out.count();

            int *lhsshape = NULL;
            cudaMalloc((void **)&lhsshape, lhs.sizes().size() * sizeof(int));

            int *rhsshape = NULL;
            cudaMalloc((void **)&rhsshape, rhs.sizes().size() * sizeof(int));

            int *lhsweight = NULL;
            cudaMalloc((void **)&lhsweight, lhs.sizes().size() * sizeof(int));

            int *rhsweight = NULL;
            cudaMalloc((void **)&rhsweight, rhs.sizes().size() * sizeof(int));

            int *outweight = NULL;
            cudaMalloc((void **)&outweight, out.sizes().size() * sizeof(int));

            cudaMemcpy((void *)lhsshape, (void *)lhs.sizes().data(), lhs.sizes().size() * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy((void *)rhsshape, (void *)rhs.sizes().data(), rhs.sizes().size() * sizeof(int), cudaMemcpyHostToDevice);

            cudaMemcpy((void *)lhsweight, (void *)lhs_hype.weight().data(), lhs_hype.weight().size() * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy((void *)rhsweight, (void *)rhs_hype.weight().data(), rhs_hype.weight().size() * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy((void *)outweight, (void *)out_hype.weight().data(), out_hype.weight().size() * sizeof(int), cudaMemcpyHostToDevice);

            reduce_operator_kernel<T> <<< CUDA_BLOCK(ncount, CUDA_THREAD_NUM), CUDA_THREAD_NUM >>> (pout, ncount,
                        plhs, prhs, lhsshape, lhsweight, rhsshape, rhsweight, outweight, out.sizes().size());

            cudaFree(lhsshape);
            cudaFree(rhsshape);

            cudaFree(lhsweight);
            cudaFree(rhsweight);
            cudaFree(outweight);


            /*
            for(int i = 0; i < ncount; i++) {
                auto &tmpshape = out_iterator.coordinate();
                reduce_operator(pout[i], plhs[to_mod_index(lhs_hype, tmpshape)], prhs[to_mod_index(rhs_hype, tmpshape)]);
                ++out_iterator;
            }
            */
        }

        /*
        template<typename T>
        inline void compute_run_scalar(const T *plhs, T scalar, T *pout, size_t count) {
            // this is CPU operator, so just using memcpy
            if (pout != plhs) std::memcpy(pout, plhs, count * sizeof(T));

            for (size_t i = 0; i < count; ++i) {
                reduce_operator(pout[i], scalar);
            }
        }

	template<>
	inline void compute_run_scalar(const float *plhs, float scalar, float *pout, size_t count) {
//#ifdef TS_USE_OPENMP
//			//#pragma omp parallel for num_threads(1)
//#pragma omp parallel for num_threads(openmp_threads(count))
//#endif
		for (int i = 0; i < count - 3; i += 4) {
			float32x4 pout_x4 = float32x4(&plhs[i]) * float32x4(scalar);
			pout_x4.store(&pout[i]);
		}
		for (int i = count / 4 * 4; i < count; ++i)
		{
			reduce_operator(pout[i], plhs[i], scalar);
		}
	}

        template<typename T>
        inline void compute_run_same_shape(const T *plhs, const T *prhs, T *pout, size_t count) {
            // this is CPU operator, so just using memcpy
            if (pout != plhs) std::memcpy(pout, plhs, count * sizeof(T));

            for (size_t i = 0; i < count;++i) {
                reduce_operator(pout[i], prhs[i]);
            }
        }

	template<>
	inline void compute_run_same_shape(const float *plhs, const float *prhs, float *pout, size_t count) {
//#ifdef TS_USE_OPENMP
//			//#pragma omp parallel for num_threads(1)
//#pragma omp parallel for num_threads(openmp_threads(count))
//#endif
		for (int i = 0; i < count - 3; i += 4) {
			float32x4 pout_x4 = float32x4(&plhs[i]) * float32x4(&prhs[i]);
			pout_x4.store(&pout[i]);
		}
		for (int i = count / 4 * 4; i < count; ++i)
		{
			reduce_operator(pout[i], plhs[i], prhs[i]);
		}
	}
        */
        template<typename T>
        static inline void mul_gpu_compute_run_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<T>();

            cudaMemcpy((void *)pout, (void *)plhs, out.count() * sizeof(T), cudaMemcpyDeviceToDevice);
            reduce_operator_scalar_kernel<T> <<< CUDA_BLOCK(out.count(), CUDA_THREAD_NUM), CUDA_THREAD_NUM >>> (pout, out.count(), prhs);

            //auto scalar = prhs[0];
            //compute_run_scalar(plhs, scalar, pout, size_t(out.count()));
        }


        template<typename T>
        static inline void mul_gpu_compute_run_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<T>();

            cudaMemcpy((void *)pout, (void *)plhs, out.count() * sizeof(T), cudaMemcpyDeviceToDevice);
            reduce_operator_same_shape_kernel<T> <<< CUDA_BLOCK(out.count(), CUDA_THREAD_NUM), CUDA_THREAD_NUM >>> (pout, prhs, out.count());

            //compute_run_same_shape(plhs, prhs, pout, size_t(out.count()));
        }


        template<typename T>
        static inline void mul_gpu_compute_run_bias(const Tensor &lhs, const Tensor &rhs, Tensor &out, int dim) {
            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<T>();

            //if (pout != plhs) std::memcpy(pout, plhs, out.count() * sizeof(T));

            auto &out_shape = out.sizes();

            auto number = std::accumulate(out_shape.begin(), out_shape.begin() + dim, 1, std::multiplies<int>());
            auto count = std::accumulate(out_shape.begin() + dim + 1, out_shape.end(), 1, std::multiplies<int>());

            auto channels = out_shape[dim];


            cudaMemcpy((void *)pout, (void *)plhs, out.count() * sizeof(T), cudaMemcpyDeviceToDevice);

            reduce_operator_bias_kernel<T> <<< CUDA_BLOCK(out.count(), CUDA_THREAD_NUM), CUDA_THREAD_NUM >>> (pout, out.count(), count, channels, prhs, rhs.count());



            /*
            if (count == 1) {
                for (int n = 0; n < number; ++n) {
                    auto pchannels = pout + n * channels;
                    auto pscalar = prhs;
                    for (int c = 0; c < channels; ++c) {
                        reduce_operator(*pchannels, *pscalar);
                        ++pchannels;
                        ++pscalar;
                    }
                }
            } else {
                for (int n = 0; n < number; ++n) {
                    for (int c = 0; c < channels; ++c) {
                        int offset = (n * channels + c) * count;
                        auto local_pout = pout + offset;
                        compute_run_scalar(local_pout, prhs[channels], local_pout, size_t(count));
                    }
                }
            }
            */
        }


        void Mul::reduce_with_broadcast(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch(dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { mul_gpu_compute_run<TYPE>(lhs, rhs, out); break; }
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
                    TS_LOG_ERROR << "mul not support this data type: " << dtype << eject;
                    break;
                }
            }
        }

        void Mul::reduce_with_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch(dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { mul_gpu_compute_run_scalar<TYPE>(lhs, rhs, out); break; }
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
                    TS_LOG_ERROR << "mul not support this data type: " << dtype << eject;
                    break;
                }
            }
        }

        void Mul::reduce_with_bias(const Tensor &lhs, const Tensor &rhs, Tensor &out, int dim) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch(dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { mul_gpu_compute_run_bias<TYPE>(lhs, rhs, out, dim); break; }
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
                    TS_LOG_ERROR << "mul not support this data type: " << dtype << eject;
                    break;
                }
            }
        }

        void Mul::reduce_with_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch(dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { mul_gpu_compute_run_same_shape<TYPE>(lhs, rhs, out); break; }
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
                    TS_LOG_ERROR << "mul not support this data type: " << dtype << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(Mul, GPU, name::layer::mul())

