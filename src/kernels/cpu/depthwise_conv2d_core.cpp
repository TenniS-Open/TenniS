#include <kernels/cpu/depthwise_conv2d_core.h>
#include <core/tensor_builder.h>
#include <backend/name.h>
#include <utils/assert.h>


namespace ts {
    namespace cpu {
        template<typename T>
        static void
        cpu_depthwise_conv2d_nchw_compute_run(const Tensor &x, const Padding2D &padding, float padding_value,
                                              const Tensor &weight, const Stride2D &stride, const Dilation2D &dilation,
                                              Tensor &out, Stack &stack, bool kernel_packed) {
            if (kernel_packed) {
                TS_LOG_ERROR << "What a Terrible Failure: dealing packed weights without pack support." << eject;
            }

            auto weight_shape = weight.sizes();
            auto output_shape = out.sizes();
            auto input_shape = x.sizes();

            const T *pinput = x.data<T>();
            const T *pweight_base = weight.data<T>();
            T *poutput = out.data<T>();
            for (int n = 0; n < output_shape[0]; n++) {
                for (int c = 0; c < output_shape[1]; c++) {
                    for (int h = 0; h < output_shape[2]; h++) {
                        for (int w = 0; w < output_shape[3]; w++) {
                            const T *pweight = pweight_base + c * weight_shape[2] * weight_shape[3];
                            T value = 0;
                            for (int kh = 0; kh < weight_shape[2]; kh++) {
                                for (int kw = 0; kw < weight_shape[3]; kw++) {
                                    int h_in = -padding.top + h * stride.height + kh * dilation.height;
                                    int w_in = -padding.left + w * stride.width + kw * dilation.width;
                                    if ((h_in >= 0) && (h_in < input_shape[2]) && (w_in >= 0) &&
                                        (w_in < input_shape[3])) {
                                        int offset =
                                                ((n * output_shape[1] + c) * input_shape[2] + h_in) * input_shape[3] +
                                                w_in;
                                        value += (*pweight) * pinput[offset];
                                    }
                                    ++pweight;
                                }
                            }
                            *poutput++ = value;
                        }
                    }
                }
            }
        }

        void
        DepthwiseConv2DCore::conv2d(const Tensor &x, const Padding2D &padding, float padding_value, const Tensor &w,
                                    const Stride2D &stride, const Dilation2D &dilation, Conv2DFormat format,
                                    Tensor &out, Stack &stack, bool kernel_packed) {
            if (format != FORMAT_NCHW) {
                TS_LOG_ERROR << "DepthwiseConv2D only support NCHW" << eject;
            }
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_depthwise_conv2d_nchw_compute_run<TYPE>(x, padding, padding_value, w, stride, dilation, out, stack, kernel_packed); break; }
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << "DepthwiseConv2D not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
        }
    }
}
