//
// Created by kier on 2019/2/19.
//

#include <kernels/cpu/pooling2d_core.h>

#include "kernels/cpu/pooling2d_core.h"

#include <algorithm>
#ifdef TS_USE_OPENMP
#include <kernels/common/openmp.h>
#endif

namespace ts {
    namespace cpu {

        template<typename T>
        static bool cpu_max_pooling(
                const T *input_data, T *output_data,
                const Shape &input_shape, const Shape &output_shape,
                const KSize2D &ksize, const Stride2D &stride,
                const Padding2D &m_padding) {
            int input_h = input_shape[2];
            int input_w = input_shape[3];
            int output_h = output_shape[2];
            int output_w = output_shape[3];
            int input_channel_size = input_h * input_w;
            int output_channel_size = output_h * output_w;
            for (int n = 0; n < output_shape[0]; n++) {
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < output_shape[1]; c++) {
                    for (int oh = 0; oh < output_shape[2]; oh++) {
                        int ihStart = oh * stride.height - m_padding.top;
                        int ihEnd = std::min<T>(ihStart + ksize.height, input_h);
                        for (int ow = 0; ow < output_shape[3]; ow++) {
                            int iwStart = ow * stride.width - m_padding.left;
                            int iwEnd = std::min<T>(iwStart + ksize.width, input_w);
                            ihStart = std::max<T>(ihStart, 0);
                            iwStart = std::max<T>(iwStart, 0);
                            int outIndex = oh * output_w + ow;
                            //T maxVlue = 0;
                            T maxVlue = input_data[ihStart * input_w + iwStart];
                            //int count = 0;
                            for (int ih = ihStart; ih < ihEnd; ih++) {
                                for (int iw = iwStart; iw < iwEnd; iw++) {
                                    int input_index = ih * input_w + iw;
                                    if (input_data[input_index] > maxVlue) {
                                        maxVlue = input_data[input_index];
                                    }
                                }
                                //count++;
                            }
                            output_data[outIndex] = maxVlue;
                            //if (count == m_kernel_h * m_kernel_w)
                            //	output_data[outIndex] = maxVlue;
                            //else
                            //	output_data[outIndex] = std::max<T>(maxVlue, padding_value);
                        }
                    }
                    input_data += input_channel_size;
                    output_data += output_channel_size;
                }
            }
            return true;
        }

        template<typename T>
        static bool cpu_average_pooling(
                const T *input_data, T *output_data,
                const Shape &input_shape, const Shape &output_shape,
                const KSize2D &ksize, const Stride2D &stride,
                const Padding2D &m_padding) {
            int input_h = input_shape[2];
            int input_w = input_shape[3];
            int output_h = output_shape[2];
            int output_w = output_shape[3];
            int input_channel_size = input_h * input_w;
            int output_channel_size = output_h * output_w;
            for (int n = 0; n < output_shape[0]; n++) {
                for (int c = 0; c < output_shape[1]; c++) {
                    for (int oh = 0; oh < output_shape[2]; oh++) {
                        int ihStart = oh * stride.height - m_padding.top;
                        int ihEnd = std::min<T>(ihStart + ksize.height, input_h);
                        for (int ow = 0; ow < output_shape[3]; ow++) {
                            int iwStart = ow * stride.width - m_padding.left;
                            int iwEnd = std::min<T>(iwStart + ksize.width, input_w);
                            ihStart = std::max<T>(ihStart, 0);
                            iwStart = std::max<T>(iwStart, 0);
                            int outIndex = oh * output_w + ow;
                            T sumValue = 0.0;
                            int count = 0;
                            for (int ih = ihStart; ih < ihEnd; ih++) {
                                for (int iw = iwStart; iw < iwEnd; iw++) {
                                    int input_index = ih * input_w + iw;
                                    sumValue += input_data[input_index];
                                    count++;
                                }
                            }
                            if (count == 0)
                                output_data[outIndex] = 0;
                            else
                                output_data[outIndex] = sumValue / count;
                            //if (count == 0)
                            //	output_data[outIndex] = 0;
                            //else if (count == m_kernel_h * m_kernel_w)
                            //	output_data[outIndex] = sumValue / count;
                            //else
                            //	output_data[outIndex] = (sumValue + (m_kernel_h * m_kernel_w - count) * padding_value) / (m_kernel_h * m_kernel_w);
                        }
                    }
                    input_data += input_channel_size;
                    output_data += output_channel_size;
                }
            }
            return true;
        }

        template<typename T>
        static bool cpu_average_pooling_white(
                const T *input_data, T *output_data,
                const Shape &input_shape, const Shape &output_shape,
                const KSize2D &ksize, const Stride2D &stride,
                const Padding2D &m_padding) {
            int input_h = input_shape[2];
            int input_w = input_shape[3];
            int output_h = output_shape[2];
            int output_w = output_shape[3];
            int input_channel_size = input_h * input_w;
            int output_channel_size = output_h * output_w;
            const int count = ksize.height * ksize.width;
            for (int n = 0; n < output_shape[0]; n++) {
                for (int c = 0; c < output_shape[1]; c++) {
                    for (int oh = 0; oh < output_shape[2]; oh++) {
                        int ihStart = oh * stride.height - m_padding.top;
                        int ihEnd = std::min<T>(ihStart + ksize.height, input_h);
                        for (int ow = 0; ow < output_shape[3]; ow++) {
                            int iwStart = ow * stride.width - m_padding.left;
                            int iwEnd = std::min<T>(iwStart + ksize.width, input_w);
                            ihStart = std::max<T>(ihStart, 0);
                            iwStart = std::max<T>(iwStart, 0);
                            int outIndex = oh * output_w + ow;
                            T sumValue = 0.0;
                            for (int ih = ihStart; ih < ihEnd; ih++) {
                                for (int iw = iwStart; iw < iwEnd; iw++) {
                                    int input_index = ih * input_w + iw;
                                    sumValue += input_data[input_index];
                                }
                            }
                            output_data[outIndex] = sumValue / count;
                        }
                    }
                    input_data += input_channel_size;
                    output_data += output_channel_size;
                }
            }
            return true;
        }

        template<typename T>
        static void cpu_pooling2d_compute_run(
                const Tensor &x, Pooling2DType type, const Padding2D &padding,
                Padding2DType padding_type, const Size2D &ksize, const Stride2D &stride,
                Conv2DFormat format, Tensor &out) {
#define ENCODE(a, b) ((int(a) << 8) | int(b))
            auto code = ENCODE(padding_type, type);
            static const auto BLACK_MAX = ENCODE(Padding2DType::BLACK, Pooling2DType::MAX);
            static const auto BLACK_AVG = ENCODE(Padding2DType::BLACK, Pooling2DType::AVG);
            static const auto WHITE_MAX = ENCODE(Padding2DType::WHITE, Pooling2DType::MAX);
            static const auto WHITE_AVG = ENCODE(Padding2DType::WHITE, Pooling2DType::AVG);

            if (code == BLACK_MAX) {
                cpu_max_pooling(x.data<T>(), out.data<T>(), x.sizes(), out.sizes(),
                                ksize, stride, padding);
            } else if (code == BLACK_AVG) {
                cpu_average_pooling(x.data<T>(), out.data<T>(), x.sizes(), out.sizes(),
                                    ksize, stride, padding);
            } else if (code == WHITE_MAX) {
                cpu_max_pooling(x.data<T>(), out.data<T>(), x.sizes(), out.sizes(),
                                    ksize, stride, padding);
            } else if (code == WHITE_AVG) {
                cpu_average_pooling_white(x.data<T>(), out.data<T>(), x.sizes(), out.sizes(),
                                          ksize, stride, padding);
            } else {
                TS_LOG_ERROR << "Pooling type only support MAX and AVG with BLACK and WHITE padding" << eject;
            }
#undef ENCODE
        }

        void Pooling2DCore::pooling2d(const Tensor &x, Pooling2DType type, const Padding2D &padding,
                                      Padding2DType padding_type, const Size2D &ksize, const Stride2D &stride,
                                      Conv2DFormat format, Tensor &out) {
            if (format != FORMAT_NCHW) {
                TS_LOG_ERROR << "Pooling2D only support NCHW" << eject;
            }
            if (padding_type != Padding2DType::BLACK && padding_type != Padding2DType::WHITE) {
                TS_LOG_ERROR << "Pooling2D only support black padding or white padding" << eject;
            }
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_pooling2d_compute_run<TYPE>(x, type, padding, padding_type, ksize, stride, format, out); break; }
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << "Pooling2D not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
        }
    }
}