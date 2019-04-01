#include <kernels/gpu/transpose.h>
#include <set>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>
#include <core/tensor_builder.h>


#include "device_launch_parameters.h"
#include <cuda_runtime.h>



namespace ts {
    namespace gpu {

        template<typename T>
        static __global__ void Transpose_transpose_run_kernel(T* out, int size, const T* input,  
                                               int *inputshape, int *inputweight,
                                               int *outshape, int *outweight,
                                               int *permute, int shapelen) {
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
            ptmp = inputweight + 1;
            int ntmp = index;
            for(i= shapelen - 1; i; --i) {
                *ptr = ntmp / *ptmp;
                ntmp %= *ptmp;
                ++ptmp;
                ++ptr;
            }

            *ptr = ntmp;

            for(i=0; i<shapelen; ++i) {
                buffer2[i] = buffer1[permute[i]];
            } 

            int outindex = 0;
            for(i=0; i<shapelen; ++i) {
                buffer1[i] = buffer2[i] % outshape[i];
            }

            for(k=0, i=1; i < shapelen; ++k,++i) {
                outindex += buffer1[k] * outweight[i];
            }
            outindex += buffer1[k];

            out[outindex] = input[index]; 

            delete [] buffer1;
            delete [] buffer2;
        }



        template<typename T>
        static void Transpose_transpose_run(
                const Tensor &x, Tensor &out, int len,
                const std::vector<int> &permute,
                const Shape &input_shape, const Shape &output_shape) {
            const T* psrc = x.data<T>();
            T* pdst = out.data<T>();
            HypeShape hype_input_shape(input_shape);
            HypeShape hype_output_shape(output_shape);

            int *input_shape_dev = nullptr;
            int *input_weight = nullptr;
            int *permute_shape_dev = nullptr;
            int *output_shape_dev = nullptr;
            int *output_weight = nullptr;

            Shape tmpshape;
            tmpshape.resize(1);
            tmpshape[0] = input_shape.size();
            Tensor input_weight_tensor(out.device(), INT32, tmpshape);
            input_weight = input_weight_tensor.data<int32_t>();

            tmpshape[0] = input_shape.size();
            Tensor input_tensor(out.device(), INT32, tmpshape);
            input_shape_dev = input_tensor.data<int32_t>();

            tmpshape[0] = permute.size();
            Tensor permute_tensor(out.device(), INT32, tmpshape);
            permute_shape_dev = permute_tensor.data<int32_t>();

            tmpshape[0] = output_shape.size();
            Tensor out_tensor(out.device(), INT32, tmpshape);
            output_shape_dev = out_tensor.data<int32_t>();

            tmpshape[0] = output_shape.size();
            Tensor out_weight_tensor(out.device(), INT32, tmpshape);
            output_weight = out_weight_tensor.data<int32_t>();


            memcpy((void*)input_shape_dev, out.device(), input_shape.size() * sizeof(int32_t),
                   (void*)input_shape.data(), MemoryDevice(CPU), input_shape.size() * sizeof(int32_t));
            memcpy((void*)input_weight, out.device(), input_shape.size() * sizeof(int32_t),
                   (void*)hype_input_shape.weight().data(), MemoryDevice(CPU), input_shape.size() * sizeof(int32_t));

            memcpy((void*)output_shape_dev, out.device(), output_shape.size() * sizeof(int32_t),
                   (void*)output_shape.data(), MemoryDevice(CPU), output_shape.size() * sizeof(int32_t));

            memcpy((void*)output_weight, out.device(), output_shape.size() * sizeof(int32_t),
                   (void*)hype_output_shape.weight().data(), MemoryDevice(CPU), input_shape.size() * sizeof(int32_t));

            memcpy((void*)permute_shape_dev, out.device(), permute.size() * sizeof(int32_t),
                   (void*)permute.data(), MemoryDevice(CPU), permute.size() * sizeof(int32_t));

            Transpose_transpose_run_kernel<T> <<< CUDA_BLOCK(len, CUDA_THREAD_NUM), CUDA_THREAD_NUM >>> (pdst, len,
                        psrc, input_shape_dev, input_weight, output_shape_dev, output_weight, permute_shape_dev, input_shape.size());

        }

        template<typename T>
        static inline void gpu_transpose_compute_run(const Tensor &x, const std::vector<int> &permute, Tensor &out) {
            Transpose_transpose_run<T>(x, out, x.count(), permute, x.sizes(), out.sizes());
        }

        void Transpose::transpose(const Tensor &x, const std::vector<int> &permute, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_transpose_compute_run<TYPE>(x, permute, out); break; }
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
                    TS_LOG_ERROR << "transpose not support this data type: " << dtype << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace gpu;
//TS_REGISTER_OPERATOR(Transpose, GPU, name::layer::transpose())
