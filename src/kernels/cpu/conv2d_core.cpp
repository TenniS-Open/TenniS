#include <kernels/cpu/conv2d_core.h>
#include <core/tensor_builder.h>
#include <kernels/cpu/math_cpu.h>
#include <kernels/cpu/im2col.h>
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
        static void cpu_conv2d_nchw_compute_run(const Tensor &x, const Padding2D &padding, float padding_value,
                                           const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                                           Tensor &out, Stack &stack) {
            auto weight_shape = w.sizes();
            auto output_shape = out.sizes();
            auto x_shape = x.sizes();
            int kernel_dims = weight_shape[1] * weight_shape[2] * weight_shape[3];
            int conv_out_spatial_dim = output_shape[2] * output_shape[3];
            int output_number_offset = output_shape[1] * conv_out_spatial_dim;
            int input_number_offset = x_shape[1] * x_shape[2] * x_shape[3];
            int col_buffer_size = x_shape[1] * weight_shape[2] * weight_shape[3] * output_shape[2] * output_shape[3];

            auto number = x_shape[0];
            auto input_channels = x_shape[1];
            Size2D ksize(weight_shape[2], weight_shape[3]);
            Size2D input(x_shape[2], x_shape[3]);

            const T *pinput = x.data<T>();
            const T *pweight = w.data<T>();
            T *poutput = out.data<T>();

            Tensor col_tensor;
            T *col_buffer = nullptr;

            bool is_1x1_conv = stride.height == 1 && stride.width == 1 &&
                               ksize.height == 1 && ksize.width == 1 &&
                               padding.top == 0 && padding.bottom == 0 &&
                               padding.left == 0 && padding.right == 0;

            // 1x1 conv do not need im2col
            if (!is_1x1_conv) {
                Shape col_shape;
                col_shape.resize(1);
                col_shape[0] = col_buffer_size;
                col_tensor = stack.make(out.dtype(), col_shape, MemoryDevice(CPU));
                col_buffer = col_tensor.data<T>();
            }

            for (int i = 0; i < number; i++) {
                if (is_1x1_conv) {
                    //std::memcpy(col_buffer,pinput,sizeof(T)*col_buffer_size);
                    col_buffer = const_cast<T *>(pinput);
                } else {
                    ::memset(col_buffer, 0, col_buffer_size * sizeof(T));
                    im2col_cpu(pinput, input_channels, input.height, input.width,
                               ksize.height, ksize.width,
                               padding.top, padding.bottom,
                               padding.left, padding.right,
                               stride.height, stride.width,
                               dilation.height, dilation.width,
                               col_buffer, T(padding_value));
                }
#ifdef TS_USE_CBLAS
                cblas::math<T>::gemm(ts::blas::NoTrans, ts::blas::NoTrans, weight_shape[0], conv_out_spatial_dim,
                                     kernel_dims, 1.0, pweight, col_buffer, 0, poutput);
#else
                cpu::math<T>::gemm(ts::blas::NoTrans,ts::blas::NoTrans, weight_shape[0], conv_out_spatial_dim,
                               kernel_dims, 1.0, pweight, col_buffer, 0, poutput);
#endif
                pinput += input_number_offset;
                poutput += output_number_offset;
            }
        }

        void Conv2DCore::conv2d(const Tensor &x, const Padding2D &padding, float padding_value, const Tensor &w,
                            const Stride2D &stride, const Dilation2D &dilation, Conv2DFormat format, Tensor &out,
                            Stack &stack) {
            if (format != FORMAT_NCHW) {
                TS_LOG_ERROR << "Conv2D only support NCHW" << eject;
            }
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_conv2d_nchw_compute_run<TYPE>(x, padding, padding_value, w, stride, dilation, out, stack);; break; }
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << "Conv2D not support this data type: " << type_str(dtype) << eject;
                    break;
                }
            }
        }
    }
}
