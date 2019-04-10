#include <kernels/opt/conv2d_core.h>
#include <core/tensor_builder.h>
#include <kernels/cpu/math_cpu.h>
#include <kernels/cpu/im2col.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

#include <global/operator_factory.h>

#include "module/bubble.h"
#include "core/tensor_builder.h"
#include <kernels/cpu/pad.h>
#include "kernels/common/simd.h"
#include "kernels/opt/conv_algorithm.h"
#ifdef TS_USE_CBLAS
#include <kernels/cblas/math_cblas.h>
#endif
#ifdef TS_USE_OPENMP
#include <omp.h>
#endif
#include <array>

namespace ts {
    namespace opt {

        using conv_function = std::function<void(const Tensor &,const Tensor &, Tensor &)>;

        //typedef void(*conv_function)(const Tensor&, const Tensor&, const Dilation2D&, Tensor &);

        //template<typename T>
        //static void opt_conv2d_dilation_forward(conv_function f, const Tensor &x, const Padding2D &padding, float padding_value,
        //    const Tensor &w, const Stride2D &stride, const Dilation2D &dilation, Tensor &out) {

        //}

        template<typename T>
        static void opt_conv2d_3x3_sse(const Tensor &x,const Tensor &w, Tensor &out){

            auto x_shape = x.sizes();
            auto out_shape = out.sizes();
            int number = x_shape[0];
            int input_channel = x_shape[1];
            int input_height = x_shape[2];
            int input_width = x_shape[3];
            int input_channel_offset = input_height *input_width;
            int input_num_offset = input_channel * input_channel_offset;

            int out_channel = out_shape[1];
            int out_height = out_shape[2];
            int out_width = out_shape[3];
    
            int out_channel_offset = out_height * out_width;
            int out_num_offset = out_channel * out_channel_offset;

            const T *pinput = x.data<T>();
            const T *pweight = w.data<T>();
            T *poutput = out.data<T>();
            std::memset(poutput, T(0), sizeof(T)*out.count());

            for (int n = 0; n < number; n++)
            {
                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (int outc_index = 0; outc_index < out_channel; outc_index++)
                {
                    T* out_at = poutput + n*out_num_offset + outc_index * out_channel_offset;

                    for (int inc_index = 0; inc_index < input_channel; inc_index++)
                    {
                        T* out_ptr_0 = out_at;
                        T* out_ptr_1 = out_ptr_0 + out_width;

                        const T* input_cur = pinput + n*input_num_offset + inc_index*input_channel_offset;

                        const T* kernel_cur = pweight + outc_index * input_channel * 9 + inc_index * 9;

                        const T* r_0 = input_cur;
                        const T* r_1 = input_cur + input_width;
                        const T* r_2 = input_cur + input_width * 2;
                        const T* r_3 = input_cur + input_width * 3;

                        const T* k_0 = kernel_cur;
                        const T* k_1 = kernel_cur + 3;
                        const T* k_2 = kernel_cur + 6;

                        int i = 0;
                        for (; i+1 < out_height; i+=2)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                T sum_0 = 0,sum_1 = 0;

                                sum_0 += r_0[0] * k_0[0];
                                sum_0 += r_0[1] * k_0[1];
                                sum_0 += r_0[2] * k_0[2];
                                sum_0 += r_1[0] * k_1[0];
                                sum_0 += r_1[1] * k_1[1];
                                sum_0 += r_1[2] * k_1[2];
                                sum_0 += r_2[0] * k_2[0];
                                sum_0 += r_2[1] * k_2[1];
                                sum_0 += r_2[2] * k_2[2];

                                sum_1 += r_1[0] * k_0[0];
                                sum_1 += r_1[1] * k_0[1];
                                sum_1 += r_1[2] * k_0[2];
                                sum_1 += r_2[0] * k_1[0];
                                sum_1 += r_2[1] * k_1[1];
                                sum_1 += r_2[2] * k_1[2];
                                sum_1 += r_3[0] * k_2[0];
                                sum_1 += r_3[1] * k_2[1];
                                sum_1 += r_3[2] * k_2[2];

                                *out_ptr_0 += sum_0;
                                *out_ptr_1 += sum_1;

                                r_0++;
                                r_1++;
                                r_2++;
                                r_3++;
                                out_ptr_0++;
                                out_ptr_1++;
                            }

                            r_0 += 2 + input_width;
                            r_1 += 2 + input_width;
                            r_2 += 2 + input_width;
                            r_3 += 2 + input_width;

                            out_ptr_0 += out_width;
                            out_ptr_1 += out_width;
                        }

                        for (; i < out_height; i++)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                T sum = 0;
                                
                                sum += r_0[0] * k_0[0];
                                sum += r_0[1] * k_0[1];
                                sum += r_0[2] * k_0[2];
                                sum += r_1[0] * k_1[0];
                                sum += r_1[1] * k_1[1];
                                sum += r_1[2] * k_1[2];
                                sum += r_2[0] * k_2[0];
                                sum += r_2[1] * k_2[1];
                                sum += r_2[2] * k_2[2];

                                *out_ptr_0 += sum;

                                r_0++;
                                r_1++;
                                r_2++;
                                out_ptr_0++;
                            }

                            r_0 += 2;
                            r_1 += 2;
                            r_2 += 2;
                        }
                    }
                }
            }
        }

        template<>
        void opt_conv2d_3x3_sse<float>(const Tensor &x, const Tensor &w, Tensor &out) {

            auto x_shape = x.sizes();
            auto out_shape = out.sizes();
            int number = x_shape[0];
            int input_channel = x_shape[1];
            int input_height = x_shape[2];
            int input_width = x_shape[3];
            int input_channel_offset = input_height *input_width;
            int input_num_offset = input_channel * input_channel_offset;

            int out_channel = out_shape[1];
            int out_height = out_shape[2];
            int out_width = out_shape[3];

            int out_channel_offset = out_height * out_width;
            int out_num_offset = out_channel * out_channel_offset;

            const float *pinput = x.data<float>();
            const float *pweight = w.data<float>();
            float *poutput = out.data<float>();
            std::memset(poutput, float(0), sizeof(float)*out.count());

            for (int n = 0; n < number; n++)
            {
                #pragma omp parallel for num_threads(omp_get_max_threads())
                //#pragma omp parallel for num_threads(8)
                for (int outc_index = 0; outc_index < out_channel; outc_index++)
                {
                    float* out_at = poutput + n*out_num_offset + outc_index * out_channel_offset;

                    for (int inc_index = 0; inc_index < input_channel; inc_index++)
                    {
                        float* out_ptr_0 = out_at;
                        float* out_ptr_1 = out_ptr_0 + out_width;

                        const float* input_cur = pinput + n*input_num_offset + inc_index*input_channel_offset;

                        const float* kernel_cur = pweight + outc_index * input_channel * 9 + inc_index * 9;

                        const float* r_0 = input_cur;
                        const float* r_1 = input_cur + input_width;
                        const float* r_2 = input_cur + input_width * 2;
                        const float* r_3 = input_cur + input_width * 3;

                        const float* k_0 = kernel_cur;
                        const float* k_1 = kernel_cur + 3;
                        const float* k_2 = kernel_cur + 6;
                        float32x4 k00(k_0);
                        float32x4 k10(k_1);
                        float32x4 k20(k_2);

                        int i = 0;
                        for (; i + 1 < out_height; i += 2)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                float32x4 r00(r_0);
                                float32x4 r10(r_1);
                                float32x4 r20(r_2);
                                float32x4 r30(r_3);

                                float32x4 sum00(float(0));

                                sum00 += r00 * k00;
                                sum00 += r10 * k10;
                                sum00 += r20 * k20;

                                float32x4 sum10(float(0));
                                sum10 += r10 * k00;
                                sum10 += r20 * k10;
                                sum10 += r30 * k20;

                                float sum0 = ts::sum(sum00, 3);
                                float sum1 = ts::sum(sum10, 3);

                                *out_ptr_0 += sum0;
                                *out_ptr_1 += sum1;

                                //sum00.store(out_ptr_0);
                                //sum10.store(out_ptr_1);

                                r_0++;
                                r_1++;
                                r_2++;
                                r_3++;
                                out_ptr_0++;
                                out_ptr_1++;
                            }

                            r_0 += 2 + input_width;
                            r_1 += 2 + input_width;
                            r_2 += 2 + input_width;
                            r_3 += 2 + input_width;

                            out_ptr_0 += out_width;
                            out_ptr_1 += out_width;
                        }

                        for (; i < out_height; i++)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                float32x4 r00(r_0);
                                float32x4 r10(r_1);
                                float32x4 r20(r_2);

                                float32x4 sum_x4(float(0));

                                sum_x4 += r00 * k00;
                                sum_x4 += r10 * k10;
                                sum_x4 += r20 * k20;

                                float sum = ts::sum(sum_x4, 3);

                                *out_ptr_0 += sum;

                                r_0++;
                                r_1++;
                                r_2++;
                                out_ptr_0++;
                            }

                            r_0 += 2;
                            r_1 += 2;
                            r_2 += 2;
                        }
                    }
                }
            }
        }

        template<typename T>
        static void opt_conv2d_3x3_sse_2(const Tensor &x, const Tensor &w, Tensor &out) {

            auto x_shape = x.sizes();
            auto out_shape = out.sizes();
            int number = x_shape[0];
            int input_channel = x_shape[1];
            int input_height = x_shape[2];
            int input_width = x_shape[3];
            int input_channel_offset = input_height *input_width;
            int input_num_offset = input_channel * input_channel_offset;

            int out_channel = out_shape[1];
            int out_height = out_shape[2];
            int out_width = out_shape[3];

            int out_channel_offset = out_height * out_width;
            int out_num_offset = out_channel * out_channel_offset;

            const T *pinput = x.data<T>();
            const T *pweight = w.data<T>();
            T *poutput = out.data<T>();
            std::memset(poutput, T(0), sizeof(T)*out.count());

            for (int n = 0; n < number; n++)
            {    
                int for_out_channel = out_channel >> 2;
                int remain_out_channel = for_out_channel << 2;

                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (int outc_index = 0; outc_index < for_out_channel; outc_index++)
                {
                    int p = outc_index * 4;
                    T *out_at = poutput + n * out_num_offset + p * out_channel_offset;
                    T *out_ptr_0 = out_at;
                    T *out_ptr_1 = out_ptr_0 + out_channel_offset;
                    T *out_ptr_2 = out_ptr_1 + out_channel_offset;
                    T *out_ptr_3 = out_ptr_2 + out_channel_offset;

                    int kernel_num_offset = input_channel * 9;
                    const T* kernel_cur = pweight + p * kernel_num_offset;
                    const T* k_0 = kernel_cur ;
                    const T* k_1 = k_0 + kernel_num_offset;
                    const T* k_2 = k_1 + kernel_num_offset;
                    const T* k_3 = k_2 + kernel_num_offset;

                    for (int inc_index = 0; inc_index < input_channel; inc_index++)
                    {
                        T* out_ptr_00 = out_ptr_0;
                        T* out_ptr_01 = out_ptr_0 + out_width;
                        T* out_ptr_10 = out_ptr_1;
                        T* out_ptr_11 = out_ptr_1 + out_width;
                        T* out_ptr_20 = out_ptr_2;
                        T* out_ptr_21 = out_ptr_2 + out_width;
                        T* out_ptr_30 = out_ptr_3;
                        T* out_ptr_31 = out_ptr_3 + out_width;

                        const T* input_cur = pinput + n * input_num_offset + inc_index * input_channel_offset;

                        const T* r_0 = input_cur;
                        const T* r_1 = r_0 + input_width;
                        const T* r_2 = r_1 + input_width;
                        const T* r_3 = r_2 + input_width;

                        int i = 0;
                        for (; i + 1 < out_height; i += 2)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                T sum_00 = 0, sum_01 = 0;
                                T sum_10 = 0, sum_11 = 0;
                                T sum_20 = 0, sum_21 = 0;
                                T sum_30 = 0, sum_31 = 0;

                                sum_00 += r_0[0] * k_0[0];
                                sum_00 += r_0[1] * k_0[1];
                                sum_00 += r_0[2] * k_0[2];
                                sum_00 += r_1[0] * k_0[3];
                                sum_00 += r_1[1] * k_0[4];
                                sum_00 += r_1[2] * k_0[5];
                                sum_00 += r_2[0] * k_0[6];
                                sum_00 += r_2[1] * k_0[7];
                                sum_00 += r_2[2] * k_0[8];

                                sum_01 += r_1[0] * k_0[0];
                                sum_01 += r_1[1] * k_0[1];
                                sum_01 += r_1[2] * k_0[2];
                                sum_01 += r_2[0] * k_0[3];
                                sum_01 += r_2[1] * k_0[4];
                                sum_01 += r_2[2] * k_0[5];
                                sum_01 += r_3[0] * k_0[6];
                                sum_01 += r_3[1] * k_0[7];
                                sum_01 += r_3[2] * k_0[8];

                                sum_10 += r_0[0] * k_1[0];
                                sum_10 += r_0[1] * k_1[1];
                                sum_10 += r_0[2] * k_1[2];
                                sum_10 += r_1[0] * k_1[3];
                                sum_10 += r_1[1] * k_1[4];
                                sum_10 += r_1[2] * k_1[5];
                                sum_10 += r_2[0] * k_1[6];
                                sum_10 += r_2[1] * k_1[7];
                                sum_10 += r_2[2] * k_1[8];

                                sum_11 += r_1[0] * k_1[0];
                                sum_11 += r_1[1] * k_1[1];
                                sum_11 += r_1[2] * k_1[2];
                                sum_11 += r_2[0] * k_1[3];
                                sum_11 += r_2[1] * k_1[4];
                                sum_11 += r_2[2] * k_1[5];
                                sum_11 += r_3[0] * k_1[6];
                                sum_11 += r_3[1] * k_1[7];
                                sum_11 += r_3[2] * k_1[8];

                                sum_20 += r_0[0] * k_2[0];
                                sum_20 += r_0[1] * k_2[1];
                                sum_20 += r_0[2] * k_2[2];
                                sum_20 += r_1[0] * k_2[3];
                                sum_20 += r_1[1] * k_2[4];
                                sum_20 += r_1[2] * k_2[5];
                                sum_20 += r_2[0] * k_2[6];
                                sum_20 += r_2[1] * k_2[7];
                                sum_20 += r_2[2] * k_2[8];

                                sum_21 += r_1[0] * k_2[0];
                                sum_21 += r_1[1] * k_2[1];
                                sum_21 += r_1[2] * k_2[2];
                                sum_21 += r_2[0] * k_2[3];
                                sum_21 += r_2[1] * k_2[4];
                                sum_21 += r_2[2] * k_2[5];
                                sum_21 += r_3[0] * k_2[6];
                                sum_21 += r_3[1] * k_2[7];
                                sum_21 += r_3[2] * k_2[8];

                                sum_30 += r_0[0] * k_3[0];
                                sum_30 += r_0[1] * k_3[1];
                                sum_30 += r_0[2] * k_3[2];
                                sum_30 += r_1[0] * k_3[3];
                                sum_30 += r_1[1] * k_3[4];
                                sum_30 += r_1[2] * k_3[5];
                                sum_30 += r_2[0] * k_3[6];
                                sum_30 += r_2[1] * k_3[7];
                                sum_30 += r_2[2] * k_3[8];

                                sum_31 += r_1[0] * k_3[0];
                                sum_31 += r_1[1] * k_3[1];
                                sum_31 += r_1[2] * k_3[2];
                                sum_31 += r_2[0] * k_3[3];
                                sum_31 += r_2[1] * k_3[4];
                                sum_31 += r_2[2] * k_3[5];
                                sum_31 += r_3[0] * k_3[6];
                                sum_31 += r_3[1] * k_3[7];
                                sum_31 += r_3[2] * k_3[8];

                                *out_ptr_00 += sum_00;
                                *out_ptr_01 += sum_01;
                                *out_ptr_10 += sum_10;
                                *out_ptr_11 += sum_11;
                                *out_ptr_20 += sum_20;
                                *out_ptr_21 += sum_21;
                                *out_ptr_30 += sum_30;
                                *out_ptr_31 += sum_31;

                                r_0++;
                                r_1++;
                                r_2++;
                                r_3++;
                                out_ptr_00++;
                                out_ptr_01++;
                                out_ptr_10++;
                                out_ptr_11++;
                                out_ptr_20++;
                                out_ptr_21++;
                                out_ptr_30++;
                                out_ptr_31++;

                            }

                            r_0 += 2 + input_width;
                            r_1 += 2 + input_width;
                            r_2 += 2 + input_width;
                            r_3 += 2 + input_width;

                            out_ptr_00 += out_width;
                            out_ptr_01 += out_width;
                            out_ptr_10 += out_width;
                            out_ptr_11 += out_width;
                            out_ptr_20 += out_width;
                            out_ptr_21 += out_width;
                            out_ptr_30 += out_width;
                            out_ptr_31 += out_width;
                        }

                        for (; i < out_height; i++)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                T sum_0 = 0, sum_1 = 0, sum_2 = 0, sum_3 = 0;

                                sum_0 += r_0[0] * k_0[0];
                                sum_0 += r_0[1] * k_0[1];
                                sum_0 += r_0[2] * k_0[2];
                                sum_0 += r_1[0] * k_0[3];
                                sum_0 += r_1[1] * k_0[4];
                                sum_0 += r_1[2] * k_0[5];
                                sum_0 += r_2[0] * k_0[6];
                                sum_0 += r_2[1] * k_0[7];
                                sum_0 += r_2[2] * k_0[8];

                                sum_1 += r_0[0] * k_1[0];
                                sum_1 += r_0[1] * k_1[1];
                                sum_1 += r_0[2] * k_1[2];
                                sum_1 += r_1[0] * k_1[3];
                                sum_1 += r_1[1] * k_1[4];
                                sum_1 += r_1[2] * k_1[5];
                                sum_1 += r_2[0] * k_1[6];
                                sum_1 += r_2[1] * k_1[7];
                                sum_1 += r_2[2] * k_1[8];

                                sum_2 += r_0[0] * k_2[0];
                                sum_2 += r_0[1] * k_2[1];
                                sum_2 += r_0[2] * k_2[2];
                                sum_2 += r_1[0] * k_2[3];
                                sum_2 += r_1[1] * k_2[4];
                                sum_2 += r_1[2] * k_2[5];
                                sum_2 += r_2[0] * k_2[6];
                                sum_2 += r_2[1] * k_2[7];
                                sum_2 += r_2[2] * k_2[8];

                                sum_3 += r_0[0] * k_3[0];
                                sum_3 += r_0[1] * k_3[1];
                                sum_3 += r_0[2] * k_3[2];
                                sum_3 += r_1[0] * k_3[3];
                                sum_3 += r_1[1] * k_3[4];
                                sum_3 += r_1[2] * k_3[5];
                                sum_3 += r_2[0] * k_3[6];
                                sum_3 += r_2[1] * k_3[7];
                                sum_3 += r_2[2] * k_3[8];

                                *out_ptr_00 += sum_0;
                                *out_ptr_10 += sum_1;
                                *out_ptr_20 += sum_2;
                                *out_ptr_30 += sum_3;

                                r_0++;
                                r_1++;
                                r_2++;
                                r_3++;
                                out_ptr_00++;
                                out_ptr_10++;
                                out_ptr_20++;
                                out_ptr_30++;
                            }

                            r_0 += 2;
                            r_1 += 2;
                            r_2 += 2;
                        }

                        k_0 += 9;
                        k_1 += 9;
                        k_2 += 9;
                        k_3 += 9;
                    }
                }

                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (int p = remain_out_channel; p < out_channel; p++)
                {
                    T *out_at = poutput + n * out_num_offset + p * out_channel_offset;

                    const T* kernel_cur = pweight + p * input_channel * 9;

                    for (int q = 0; q < input_channel; q++)
                    {
                        T* out_ptr_0 = out_at;
                        T* out_ptr_1 = out_ptr_0 + out_width;

                        const T* input_cur = pinput + n * input_num_offset + q * input_channel_offset;

                        const T* r_0 = input_cur;
                        const T* r_1 = r_0 + input_width;
                        const T* r_2 = r_1 + input_width;
                        const T* r_3 = r_2 + input_width;

                        const T* k_0 = kernel_cur;
                        const T* k_1 = kernel_cur + 3;
                        const T* k_2 = kernel_cur + 6;

                        int i = 0;
                        for (; i + 1 < out_height; i += 2)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                T sum_0 = 0, sum_1 = 0;

                                sum_0 += r_0[0] * k_0[0];
                                sum_0 += r_0[1] * k_0[1];
                                sum_0 += r_0[2] * k_0[2];
                                sum_0 += r_1[0] * k_1[0];
                                sum_0 += r_1[1] * k_1[1];
                                sum_0 += r_1[2] * k_1[2];
                                sum_0 += r_2[0] * k_2[0];
                                sum_0 += r_2[1] * k_2[1];
                                sum_0 += r_2[2] * k_2[2];

                                sum_1 += r_1[0] * k_0[0];
                                sum_1 += r_1[1] * k_0[1];
                                sum_1 += r_1[2] * k_0[2];
                                sum_1 += r_2[0] * k_1[0];
                                sum_1 += r_2[1] * k_1[1];
                                sum_1 += r_2[2] * k_1[2];
                                sum_1 += r_3[0] * k_2[0];
                                sum_1 += r_3[1] * k_2[1];
                                sum_1 += r_3[2] * k_2[2];

                                *out_ptr_0 += sum_0;
                                *out_ptr_1 += sum_1;

                                r_0++;
                                r_1++;
                                r_2++;
                                r_3++;
                                out_ptr_0++;
                                out_ptr_1++;
                            }

                            r_0 += 2 + input_width;
                            r_1 += 2 + input_width;
                            r_2 += 2 + input_width;
                            r_3 += 2 + input_width;

                            out_ptr_0 += out_width;
                            out_ptr_1 += out_width;
                        }

                        for (; i < out_height; i++)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                T sum = 0;

                                sum += r_0[0] * k_0[0];
                                sum += r_0[1] * k_0[1];
                                sum += r_0[2] * k_0[2];
                                sum += r_1[0] * k_1[0];
                                sum += r_1[1] * k_1[1];
                                sum += r_1[2] * k_1[2];
                                sum += r_2[0] * k_2[0];
                                sum += r_2[1] * k_2[1];
                                sum += r_2[2] * k_2[2];

                                *out_ptr_0 += sum;

                                r_0++;
                                r_1++;
                                r_2++;
                                out_ptr_0++;
                            }

                            r_0 += 2;
                            r_1 += 2;
                            r_2 += 2;
                        }

                        kernel_cur += 9;
                    }
                }
            }
        }

        template<>
        void opt_conv2d_3x3_sse_2<float>(const Tensor &x, const Tensor &w, Tensor &out) {

            auto x_shape = x.sizes();
            auto out_shape = out.sizes();
            int number = x_shape[0];
            int input_channel = x_shape[1];
            int input_height = x_shape[2];
            int input_width = x_shape[3];
            int input_channel_offset = input_height *input_width;
            int input_num_offset = input_channel * input_channel_offset;

            int out_channel = out_shape[1];
            int out_height = out_shape[2];
            int out_width = out_shape[3];

            int out_channel_offset = out_height * out_width;
            int out_num_offset = out_channel * out_channel_offset;

            const float *pinput = x.data<float>();
            const float *pweight = w.data<float>();
            float *poutput = out.data<float>();
            std::memset(poutput, float(0), sizeof(float)*out.count());

            for (int n = 0; n < number; n++)
            {
                int for_out_channel = out_channel >> 2;
                int remain_out_channel = for_out_channel << 2;

                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (int outc_index = 0; outc_index < for_out_channel; outc_index++)
                {
                    int p = outc_index * 4;
                    float *out_at = poutput + n * out_num_offset + p * out_channel_offset;
                    float *out_ptr_0 = out_at;
                    float *out_ptr_1 = out_ptr_0 + out_channel_offset;
                    float *out_ptr_2 = out_ptr_1 + out_channel_offset;
                    float *out_ptr_3 = out_ptr_2 + out_channel_offset;

                    int kernel_num_offset = input_channel * 9;
                    const float* kernel_cur = pweight + p * kernel_num_offset;
                    const float* k_0 = kernel_cur;
                    const float* k_1 = k_0 + kernel_num_offset;
                    const float* k_2 = k_1 + kernel_num_offset;
                    const float* k_3 = k_2 + kernel_num_offset;

                    for (int inc_index = 0; inc_index < input_channel; inc_index++)
                    {
                        float32x4 k00(k_0);
                        float32x4 k01(k_0 + 3);
                        float32x4 k02(k_0 + 6);

                        float32x4 k10(k_1);
                        float32x4 k11(k_1 + 3);
                        float32x4 k12(k_1 + 6);

                        float32x4 k20(k_2);
                        float32x4 k21(k_2 + 3);
                        float32x4 k22(k_2 + 6);

                        float32x4 k30(k_3);
                        float32x4 k31(k_3 + 3);
                        float32x4 k32(k_3 + 6);

                        float* out_ptr_00 = out_ptr_0;
                        float* out_ptr_01 = out_ptr_0 + out_width;
                        float* out_ptr_10 = out_ptr_1;
                        float* out_ptr_11 = out_ptr_1 + out_width;
                        float* out_ptr_20 = out_ptr_2;
                        float* out_ptr_21 = out_ptr_2 + out_width;
                        float* out_ptr_30 = out_ptr_3;
                        float* out_ptr_31 = out_ptr_3 + out_width;

                        const float* input_cur = pinput + n * input_num_offset + inc_index * input_channel_offset;

                        const float* r_0 = input_cur;
                        const float* r_1 = r_0 + input_width;
                        const float* r_2 = r_1 + input_width;
                        const float* r_3 = r_2 + input_width;

                        int i = 0;
                        for (; i + 1 < out_height; i += 2)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                float32x4 r00(r_0);
                                float32x4 r01(r_1);
                                float32x4 r02(r_2);
                                float32x4 r03(r_3);

                                float32x4 sum_00_x4(float(0)); float32x4 sum_01_x4(float(0));
                                float32x4 sum_10_x4(float(0)); float32x4 sum_11_x4(float(0));
                                float32x4 sum_20_x4(float(0)); float32x4 sum_21_x4(float(0));
                                float32x4 sum_30_x4(float(0)); float32x4 sum_31_x4(float(0));

                                sum_00_x4 += r00 * k00;
                                sum_00_x4 += r01 * k01;
                                sum_00_x4 += r02 * k02;

                                sum_01_x4 += r01 * k00;
                                sum_01_x4 += r02 * k01;
                                sum_01_x4 += r03 * k02;

                                sum_10_x4 += r00 * k10;
                                sum_10_x4 += r01 * k11;
                                sum_10_x4 += r02 * k12;

                                sum_11_x4 += r01 * k10;
                                sum_11_x4 += r02 * k11;
                                sum_11_x4 += r03 * k12;

                                sum_20_x4 += r00 * k20;
                                sum_20_x4 += r01 * k21;
                                sum_20_x4 += r02 * k22;

                                sum_21_x4 += r01 * k20;
                                sum_21_x4 += r02 * k21;
                                sum_21_x4 += r03 * k22;

                                sum_30_x4 += r00 * k30;
                                sum_30_x4 += r01 * k31;
                                sum_30_x4 += r02 * k32;

                                sum_31_x4 += r01 * k30;
                                sum_31_x4 += r02 * k31;
                                sum_31_x4 += r03 * k32;

                                *out_ptr_00 += ts::sum(sum_00_x4, 3);
                                *out_ptr_01 += ts::sum(sum_01_x4, 3);
                                *out_ptr_10 += ts::sum(sum_10_x4, 3);
                                *out_ptr_11 += ts::sum(sum_11_x4, 3);
                                *out_ptr_20 += ts::sum(sum_20_x4, 3);
                                *out_ptr_21 += ts::sum(sum_21_x4, 3);
                                *out_ptr_30 += ts::sum(sum_30_x4, 3);
                                *out_ptr_31 += ts::sum(sum_31_x4, 3);

                                r_0++;
                                r_1++;
                                r_2++;
                                r_3++;
                                out_ptr_00++;
                                out_ptr_01++;
                                out_ptr_10++;
                                out_ptr_11++;
                                out_ptr_20++;
                                out_ptr_21++;
                                out_ptr_30++;
                                out_ptr_31++;

                            }

                            r_0 += 2 + input_width;
                            r_1 += 2 + input_width;
                            r_2 += 2 + input_width;
                            r_3 += 2 + input_width;

                            out_ptr_00 += out_width;
                            out_ptr_01 += out_width;
                            out_ptr_10 += out_width;
                            out_ptr_11 += out_width;
                            out_ptr_20 += out_width;
                            out_ptr_21 += out_width;
                            out_ptr_30 += out_width;
                            out_ptr_31 += out_width;
                        }

                        for (; i < out_height; i++)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                float32x4 r00(r_0);
                                float32x4 r01(r_1);
                                float32x4 r02(r_2);
                                float32x4 r03(r_3);

                                float32x4 sum_00(float(0));
                                float32x4 sum_10(float(0));
                                float32x4 sum_20(float(0));
                                float32x4 sum_30(float(0));

                                sum_00 += r00 * k00;
                                sum_00 += r01 * k01;
                                sum_00 += r02 * k02;

                                sum_10 += r00 * k10;
                                sum_10 += r01 * k11;
                                sum_10 += r02 * k12;

                                sum_20 += r00 * k20;
                                sum_20 += r01 * k21;
                                sum_20 += r02 * k22;

                                sum_30 += r00 * k30;
                                sum_30 += r01 * k31;
                                sum_30 += r02 * k32;

                                *out_ptr_00 += ts::sum(sum_00, 3);
                                *out_ptr_10 += ts::sum(sum_10, 3);
                                *out_ptr_20 += ts::sum(sum_20, 3);
                                *out_ptr_30 += ts::sum(sum_30, 3);

                                r_0++;
                                r_1++;
                                r_2++;
                                r_3++;
                                out_ptr_00++;
                                out_ptr_10++;
                                out_ptr_20++;
                                out_ptr_30++;
                            }

                            r_0 += 2;
                            r_1 += 2;
                            r_2 += 2;
                        }

                        k_0 += 9;
                        k_1 += 9;
                        k_2 += 9;
                        k_3 += 9;
                    }
                }

                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (int p = remain_out_channel; p < out_channel; p++)
                {
                    float *out_at = poutput + n * out_num_offset + p * out_channel_offset;

                    const float* kernel_cur = pweight + p * input_channel * 9;

                    for (int q = 0; q < input_channel; q++)
                    {
                        float* out_ptr_0 = out_at;
                        float* out_ptr_1 = out_ptr_0 + out_width;

                        const float* input_cur = pinput + n * input_num_offset + q * input_channel_offset;

                        const float* r_0 = input_cur;
                        const float* r_1 = r_0 + input_width;
                        const float* r_2 = r_1 + input_width;
                        const float* r_3 = r_2 + input_width;

                        const float* k_0 = kernel_cur;
                        const float* k_1 = kernel_cur + 3;
                        const float* k_2 = kernel_cur + 6;

                        float32x4 k00(k_0);
                        float32x4 k10(k_1);
                        float32x4 k20(k_2);

                        int i = 0;
                        for (; i + 1 < out_height; i += 2)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                float32x4 r00(r_0);
                                float32x4 r10(r_1);
                                float32x4 r20(r_2);
                                float32x4 r30(r_3);

                                float32x4 sum00(float(0));
                                float32x4 sum10(float(0));

                                sum00 += r00 * k00;
                                sum00 += r10 * k10;
                                sum00 += r20 * k20;

                                sum10 += r10 * k00;
                                sum10 += r20 * k10;
                                sum10 += r30 * k20;

                                float sum_0 = ts::sum(sum00, 3);
                                float sum_1 = ts::sum(sum10, 3);

                                *out_ptr_0 += sum_0;
                                *out_ptr_1 += sum_1;

                                r_0++;
                                r_1++;
                                r_2++;
                                r_3++;
                                out_ptr_0++;
                                out_ptr_1++;
                            }

                            r_0 += 2 + input_width;
                            r_1 += 2 + input_width;
                            r_2 += 2 + input_width;
                            r_3 += 2 + input_width;

                            out_ptr_0 += out_width;
                            out_ptr_1 += out_width;
                        }

                        for (; i < out_height; i++)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                float32x4 r00(r_0);
                                float32x4 r10(r_1);
                                float32x4 r20(r_2);
                                float32x4 r30(r_3);

                                float32x4 sum00(float(0));

                                sum00 += r00 * k00;
                                sum00 += r00 * k10;
                                sum00 += r00 * k20;

                                float sum = ts::sum(sum00, 3);

                                *out_ptr_0 += sum;

                                r_0++;
                                r_1++;
                                r_2++;
                                out_ptr_0++;
                            }

                            r_0 += 2;
                            r_1 += 2;
                            r_2 += 2;
                        }

                        kernel_cur += 9;
                    }
                }
            }
        }

        template<typename T>
        static void opt_conv2d_5x5_sse(const Tensor &x, const Tensor &w, Tensor &out){
            return;
        }

        template<typename T>
        static void opt_conv2d_slide_window(const Tensor &x, const Tensor &w,
            const Stride2D &stride, const Dilation2D &dilation, Tensor &out) {
            auto weight_shape = w.sizes();
            auto output_shape = out.sizes();
            auto x_shape = x.sizes();

            int input_channel = x_shape[1];
            int num = output_shape[0];
            int out_channel = output_shape[1];
            int out_height = output_shape[2];
            int out_width = output_shape[3];

            int out_channel_offset = out_height * out_width;
            int out_num_offset = out_channel * out_channel_offset;

            int input_h = x_shape[2];
            int input_w = x_shape[3];
            int input_channel_offset = input_h * input_w;
            int input_num_offset = x_shape[1] * input_channel_offset;

            int weight_h = weight_shape[2];
            int weight_w = weight_shape[3];
            int weight_channel_offset = weight_h * weight_w;
            int weight_num_offset = weight_shape[1] * weight_channel_offset;

            const T* pinput = x.data<T>();
            const T* pweight = w.data<T>();
            T* poutput = out.data<T>();

            std::vector<int> _kernel_offset(weight_channel_offset);
            int *kernel_offset = &_kernel_offset[0];
            {
                int p1 = 0;
                int p2 = 0;
                int gap = x_shape[3] * dilation.height - weight_shape[3] * dilation.width;
                for (int h = 0; h < weight_h; h++)
                {
                    for (int w = 0; w < weight_w; w++)
                    {
                        kernel_offset[p1] = p2;
                        p1++;
                        p2 += dilation.width;
                    }
                    p2 += gap;
                }
            }

            for (int n = 0; n < num; n++)
            {
                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (int c = 0; c < out_channel; c++)
                {
                    T* output_at = poutput + n * out_num_offset + c * out_channel_offset;

                    for (int h = 0; h < out_height; h++)
                    {
                        for (int w = 0; w < out_width; w++)
                        {
                            T sum = 0;

                            const T* weight_at = pweight + c * weight_num_offset;
                            for (int c_in = 0; c_in < input_channel; c_in++)
                            {
                                const T* input_at = pinput + n * input_num_offset + c_in * input_channel_offset;
                                const T* input_cur = input_at + input_w * h * stride.height + w * stride.width;
                                for (int k = 0; k < weight_channel_offset; k++)
                                {
                                    T val = input_cur[kernel_offset[k]];
                                    T w = weight_at[k];
                                    sum += val * w;
                                }
                                weight_at += weight_channel_offset;
                            }
                            output_at[w] = sum;
                        }
                        output_at += out_width;
                    }
                }
            }
        }

        template<typename T>
        static void opt_conv2d_forward(const Tensor &x, const Padding2D &padding, float padding_value,
            const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
            Tensor &out, Stack &stack) {
            auto x_shape = x.sizes();
            auto weight_shape = w.sizes();
            Size2D ksize(weight_shape[2], weight_shape[3]);
            if (ksize.height != ksize.width || stride.height != stride.width)
                return opt_conv2d_nchw_compute_run<T>(x, padding, padding_value, w, stride, dilation, out, stack);

            int kernel_size = ksize.height;
            int stride_size = stride.height;

            if(kernel_size > 5 || stride_size > 1 || dilation.height != dilation.width)
                return opt_conv2d_nchw_compute_run<T>(x, padding, padding_value, w, stride, dilation, out, stack);

            int dilation_size = dilation.height;

            conv_function conv_func_table[5][4] = {
                {
                    0,0,0,0
                },
                {
                    0,0,0,0
                },
                {
                    opt_conv2d_3x3_sse_2<T>,
                    0,
                    0,
                    0
                }, //kernel_size = 3
                {
                    0,0,0,0
                },
                {
                    0,
                    //opt_conv2d_5x5_sse<T>,
                    0,
                    0,
                    0
                } //kernel_size = 5
            };

            conv_function conv_f = conv_func_table[kernel_size-1][stride_size-1];
            if(!conv_f)
                return opt_conv2d_nchw_compute_run<T>(x, padding, padding_value, w, stride, dilation, out, stack);

            if(dilation_size != 1)
                return opt_conv2d_nchw_compute_run<T>(x, padding, padding_value, w, stride, dilation, out, stack);
                //return opt_conv2d_dilation_forward<T>(conv_f, x, padding, padding_value, w, stride, dilation, out);

            if (padding.bottom != 0 || padding.top != 0 || padding.left != 0 || padding.right != 0) {
                auto pad_operator = OperatorCreator::Query(CPU, name::layer::pad())();
                TS_CHECK_NQ(pad_operator, nullptr) << "Can not find operator: " << name::layer::pad();
                std::array<int, 2> pad_n = { 0,0 };
                std::array<int, 2> pad_c = { 0,0 };
                std::array<int, 2> pad_h = { padding.top ,padding.bottom};
                std::array<int, 2> pad_w = { padding.left, padding.right};
                std::vector<std::array<int, 2>> pad_vec = { pad_n,pad_c, pad_h,pad_w };
                ts::Shape x_pad_shape = { x_shape[0],x_shape[1],x_shape[2] + padding.top + padding.bottom,x_shape[3] + padding.left + padding.right };
                Tensor x_pad_tensor(MemoryDevice(CPU), x.dtype(), x_pad_shape);
                //auto x_pad_tensor = stack.make(x.dtype(), x_pad_shape, MemoryDevice(CPU));
                reinterpret_cast<cpu::Pad*>(pad_operator.get())->pad(x, pad_vec, padding_value, x_pad_tensor);
                //Shape kernel_trans_shape = { weight_shape[0],weight_shape[1],8,8 };
                //Tensor kernel_trans(FLOAT32, kernel_trans_shape);
                //Conv2dAlgorithm<T>::conv3x3_winograd63_transform_kernel_1(w, kernel_trans);
                //return Conv2dAlgorithm<T>::conv3x3_winograd63(x_pad_tensor,kernel_trans,out);
                return conv_f(x_pad_tensor, w, out);

/*                ts::Shape pad_shape = { 4,2 };
                auto input_pad_param = tensor::build(INT32, pad_shape, { 0, 0, 0, 0, padding.top, padding.bottom, padding.left, padding.right });

                pad_operator->set(name::padding_value, tensor_builder<float>::build(&padding_value, 1));
                pad_operator->init();
                stack.push(x);
                stack.push(input_pad_param);
                RunOperator(pad_operator, stack, 2);
                auto x_pad_tensor = *(stack.top());
                stack.pop();
                return conv_f(x_pad_tensor, w, out);*/         
            }
            //Shape kernel_trans_shape = { weight_shape[0],weight_shape[1],8,8 };
            //Tensor kernel_trans(FLOAT32, kernel_trans_shape);
            //Conv2dAlgorithm<T>::conv3x3_winograd63_transform_kernel_1(w, kernel_trans);
            //return Conv2dAlgorithm<T>::conv3x3_winograd63(x, kernel_trans, out);
            return conv_f(x, w, out);
        }

        template<typename T>
        static void opt_conv2d_nchw_compute_run(const Tensor &x, const Padding2D &padding, float padding_value,
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

//            if (is_1x1_conv) {
//                for (int i = 0; i < number; i++) {
//                    col_buffer = const_cast<T *>(pinput);
//#ifdef TS_USE_CBLAS
//                    cblas::math<T>::gemm(ts::blas::NoTrans, ts::blas::NoTrans, weight_shape[0], conv_out_spatial_dim,
//                        kernel_dims, 1.0, pweight, col_buffer, 0, poutput);
//#else
//                    cpu::math<T>::gemm(ts::blas::NoTrans, ts::blas::NoTrans, weight_shape[0], conv_out_spatial_dim,
//                        kernel_dims, 1.0, pweight, col_buffer, 0, poutput);
//#endif
//                    pinput += input_number_offset;
//                    poutput += output_number_offset;
//                }
//            }
//            else {
//                if (padding.bottom != 0 || padding.top != 0 || padding.left != 0 || padding.right != 0) {
//                    //auto &context = ctx::ref<DeviceContext>();
//                    //auto pad_operator = OperatorCreator::Create(context.computing_device.type(), name::layer::pad(), false);
//                    auto& pad_operator = OperatorCreator::Query(CPU, name::layer::pad())();
//                    TS_CHECK_NQ(pad_operator, nullptr) << "Can not find operator: " << name::layer::pad();
//               
//                    std::array<int, 2> pad_n = { 0,0 };
//                    std::array<int, 2> pad_c = { 0,0 };
//                    std::array<int, 2> pad_h = { padding.top ,padding.bottom};
//                    std::array<int, 2> pad_w = { padding.left, padding.right};
//                    std::vector<std::array<int, 2>> pad_vec = { pad_n,pad_c, pad_h,pad_w };
//                    ts::Shape x_pad_shape = { x_shape[0],x_shape[1],x_shape[2] + padding.top + padding.bottom,x_shape[3] + padding.left + padding.right };
//                    Tensor x_pad_tensor(MemoryDevice(CPU), x.dtype(), x_pad_shape);
//                    //auto x_pad_tensor = stack.make(x.dtype(), x_pad_shape, MemoryDevice(CPU));
//                    //reinterpret_cast<cpu::Pad*>(pad_operator.get())->pad(x, pad_vec, padding_value, x_pad_tensor);
//
//                    opt_conv2d_slide_window<T>(x_pad_tensor,w,stride,dilation,out);
//
//                }
//                else {
//                    opt_conv2d_slide_window<T>(x, w, stride, dilation, out);
//                }
//            }

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
        case DTYPE: { opt_conv2d_forward<TYPE>(x, padding, padding_value, w, stride, dilation, out, stack); break; }
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
