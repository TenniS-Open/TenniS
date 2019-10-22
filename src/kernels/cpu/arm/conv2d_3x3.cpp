//
// Created by yang on 2019/10/11.
//

#include "kernels/cpu/arm/conv2d_3x3.h"
#include "kernels/cpu/pad2d_algorithm.h"
#include "kernels/common/simd.h"
#include "kernels/common/function.h"
#ifdef TS_USE_OPENMP
#include "kernels/common/openmp.h"
#endif
#include <array>

namespace ts{
namespace cpu{
namespace arm {


    template<typename T>
    void Conv2d3x3<T>::conv2d_3x3_s1(const Tensor &x,
                                     const Padding2D &padding,
                                     float padding_value,
                                     const Tensor &w,
                                     Tensor &out) {

    }

    template <>
    void Conv2d3x3<float>::conv2d_3x3_s1(const Tensor &x,
                                         const Padding2D &padding,
                                         float padding_value,
                                         const Tensor &w,
                                         Tensor &out) {

        auto kernel_shape = w.sizes();

        Tensor input_padded = x;
        Tensor out_padded = out;
        bool out_padded_flag = false;
        Stride2D stride(1, 1);
        KSize2D ksize(3, 3);
        KernelCommonFunc<float>::in_out_pad_and_fix_size(x,
                                       kernel_shape,
                                       out,
                                       2,
                                       4,
                                       padding,
                                       padding_value,
                                       stride,
                                       ksize,
                                       input_padded,
                                       out_padded,
                                       out_padded_flag);

        auto output_shape = out_padded.sizes();
        auto input_shape = input_padded.sizes();
        auto src_output_shape = out.sizes();

        const float *kernel_ptr = w.data<float>();
        const float *input_ptr = input_padded.data<float>();
        float *out_ptr = out_padded.data<float>();

        int num = input_shape[0];
        int input_channel = input_shape[1];
        int input_height = input_shape[2];
        int input_width = input_shape[3];
        int out_channel = output_shape[1];
        int out_height = output_shape[2];
        int out_width = output_shape[3];
        int out_channel_offset = out_height * out_width;
        int output_number_offset = out_channel * out_channel_offset;
        int input_channel_offset = input_height * input_width;
        int input_number_offset = input_channel * input_channel_offset;

        for (int n = 0; n < num; ++n) {
            int out_channel_bound = out_channel >> 1;
            int remain_out_channel = out_channel_bound << 1;
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int m = 0; m < out_channel_bound; ++m) {
                int mm = m * 2;
                float *out_at = out_ptr + n * output_number_offset + mm * out_channel_offset;
                float *out_ptr_0 = out_at;
                float *out_ptr_1 = out_ptr_0 + out_channel_offset;
                for (int c = 0; c < input_channel; ++c) {
                    const float *kernel_at = kernel_ptr + mm * input_channel * 9 + c * 9;
                    const float *kernel_at_0 = kernel_at;
                    const float *kernel_at_1 = kernel_at_0 + input_channel * 9;

                    const float *input_at = input_ptr + n * input_number_offset + c * input_channel_offset;
                    const float *input_at_0 = input_at;
                    const float *input_at_1 = input_at_0 + input_width;
                    const float *input_at_2 = input_at_1 + input_width;
                    const float *input_at_3 = input_at_2 + input_width;

                    float *out_at_0 = out_ptr_0;
                    float *out_at_1 = out_ptr_1;

                    float32x4 k00_x4(kernel_at_0), k01_x4(kernel_at_0 + 3), k02_x4(kernel_at_0 + 6);
                    float32x4 k10_x4(kernel_at_1), k11_x4(kernel_at_1 + 3), k12_x4(kernel_at_1 + 6);

                    for (int h = 0; h + 1 < out_height; h += 2) {
                        for (int w = 0; w + 3 < out_width; w += 4) {
//#ifdef TS_USE_NEON
//                                float32x4 i00_x4(input_at_0);
//                                float32x4 out00_x4(input_at_0 + 4);//reuse out00_x4->i0n_x4
//                                float32x4 i10_x4(input_at_1);
//                                float32x4 out01_x4(input_at_1 + 4);
//                                float32x4 i20_x4(input_at_2);
//                                float32x4 out10_x4(input_at_2 + 4);
//                                float32x4 i30_x4(input_at_3);
//                                float32x4 out11_x4(input_at_3 + 4);
//
//                                float32x4 i01_x4 = concat(i00_x4, out00_x4, 1);
//                                float32x4 i02_x4 = concat(i00_x4, out00_x4, 2);
//                                float32x4 i11_x4 = concat(i10_x4, out01_x4, 1);
//                                float32x4 i12_x4 = concat(i10_x4, out01_x4, 2);
//                                float32x4 i21_x4 = concat(i20_x4, out10_x4, 1);
//                                float32x4 i22_x4 = concat(i20_x4, out10_x4, 2);
//                                float32x4 i31_x4 = concat(i30_x4, out11_x4, 1);
//                                float32x4 i32_x4 = concat(i30_x4, out11_x4, 2);
//
//                                out00_x4 = out_at_0;out01_x4 = out_at_0 + out_width;
//                                out10_x4 = out_at_1;out11_x4 = out_at_1 + out_width;
//#else
                            float32x4 i00_x4(input_at_0), i01_x4(input_at_0 + 1), i02_x4(input_at_0 + 2);
                            float32x4 i10_x4(input_at_1), i11_x4(input_at_1 + 1), i12_x4(input_at_1 + 2);
                            float32x4 i20_x4(input_at_2), i21_x4(input_at_2 + 1), i22_x4(input_at_2 + 2);
                            float32x4 i30_x4(input_at_3), i31_x4(input_at_3 + 1), i32_x4(input_at_3 + 2);

                            float32x4 out00_x4(out_at_0), out01_x4(out_at_0 + out_width);
                            float32x4 out10_x4(out_at_1), out11_x4(out_at_1 + out_width);
//#endif
                            out00_x4 = fmadd(i00_x4, k00_x4, out00_x4, 0);
                            out00_x4 = fmadd(i01_x4, k00_x4, out00_x4, 1);
                            out00_x4 = fmadd(i02_x4, k00_x4, out00_x4, 2);
                            out00_x4 = fmadd(i10_x4, k01_x4, out00_x4, 0);
                            out00_x4 = fmadd(i11_x4, k01_x4, out00_x4, 1);
                            out00_x4 = fmadd(i12_x4, k01_x4, out00_x4, 2);
                            out00_x4 = fmadd(i20_x4, k02_x4, out00_x4, 0);
                            out00_x4 = fmadd(i21_x4, k02_x4, out00_x4, 1);
                            out00_x4 = fmadd(i22_x4, k02_x4, out00_x4, 2);

                            out01_x4 = fmadd(i10_x4, k00_x4, out01_x4, 0);
                            out01_x4 = fmadd(i11_x4, k00_x4, out01_x4, 1);
                            out01_x4 = fmadd(i12_x4, k00_x4, out01_x4, 2);
                            out01_x4 = fmadd(i20_x4, k01_x4, out01_x4, 0);
                            out01_x4 = fmadd(i21_x4, k01_x4, out01_x4, 1);
                            out01_x4 = fmadd(i22_x4, k01_x4, out01_x4, 2);
                            out01_x4 = fmadd(i30_x4, k02_x4, out01_x4, 0);
                            out01_x4 = fmadd(i31_x4, k02_x4, out01_x4, 1);
                            out01_x4 = fmadd(i32_x4, k02_x4, out01_x4, 2);

                            out10_x4 = fmadd(i00_x4, k10_x4, out10_x4, 0);
                            out10_x4 = fmadd(i01_x4, k10_x4, out10_x4, 1);
                            out10_x4 = fmadd(i02_x4, k10_x4, out10_x4, 2);
                            out10_x4 = fmadd(i10_x4, k11_x4, out10_x4, 0);
                            out10_x4 = fmadd(i11_x4, k11_x4, out10_x4, 1);
                            out10_x4 = fmadd(i12_x4, k11_x4, out10_x4, 2);
                            out10_x4 = fmadd(i20_x4, k12_x4, out10_x4, 0);
                            out10_x4 = fmadd(i21_x4, k12_x4, out10_x4, 1);
                            out10_x4 = fmadd(i22_x4, k12_x4, out10_x4, 2);

                            out11_x4 = fmadd(i10_x4, k10_x4, out11_x4, 0);
                            out11_x4 = fmadd(i11_x4, k10_x4, out11_x4, 1);
                            out11_x4 = fmadd(i12_x4, k10_x4, out11_x4, 2);
                            out11_x4 = fmadd(i20_x4, k11_x4, out11_x4, 0);
                            out11_x4 = fmadd(i21_x4, k11_x4, out11_x4, 1);
                            out11_x4 = fmadd(i22_x4, k11_x4, out11_x4, 2);
                            out11_x4 = fmadd(i30_x4, k12_x4, out11_x4, 0);
                            out11_x4 = fmadd(i31_x4, k12_x4, out11_x4, 1);
                            out11_x4 = fmadd(i32_x4, k12_x4, out11_x4, 2);

                            out00_x4.store(out_at_0);
                            out01_x4.store(out_at_0 + out_width);
                            out10_x4.store(out_at_1);
                            out11_x4.store(out_at_1 + out_width);

                            input_at_0 += 4;
                            input_at_1 += 4;
                            input_at_2 += 4;
                            input_at_3 += 4;

                            out_at_0 += 4;
                            out_at_1 += 4;
                        } //out_w
                        input_at_0 += 2 + input_width;
                        input_at_1 += 2 + input_width;
                        input_at_2 += 2 + input_width;
                        input_at_3 += 2 + input_width;

                        out_at_0 += out_width;
                        out_at_1 += out_width;
                    } //out_h
                } //c
            } //m
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int m = remain_out_channel; m < out_channel; ++m) {
                float *out_at = out_ptr + n * output_number_offset + m * out_channel_offset;
                float *out_ptr_0 = out_at;

                for (int c = 0; c < input_channel; ++c) {
                    const float *kernel_at = kernel_ptr + m * input_channel * 9 + c * 9;
                    const float *kernel_at_0 = kernel_at;

                    const float *input_at = input_ptr + n * input_number_offset + c * input_channel_offset;
                    const float *input_at_0 = input_at;
                    const float *input_at_1 = input_at_0 + input_width;
                    const float *input_at_2 = input_at_1 + input_width;
                    const float *input_at_3 = input_at_2 + input_width;

                    float *out_at_0 = out_ptr_0;

                    float32x4 k00_x4(kernel_at_0), k01_x4(kernel_at_0 + 3), k02_x4(kernel_at_0 + 6);

                    for (int h = 0; h + 1 < out_height; h += 2) {
                        for (int w = 0; w + 3 < out_width; w += 4) {
//#ifdef TS_USE_NEON
//                                float32x4 i00_x4(input_at_0);
//                                float32x4 i0n_x4(input_at_0 + 4);
//                                float32x4 i10_x4(input_at_1);
//                                float32x4 i1n_x4(input_at_1 + 4);
//                                float32x4 i20_x4(input_at_2);
//                                float32x4 i2n_x4(input_at_2 + 4);
//                                float32x4 i30_x4(input_at_3);
//                                float32x4 i3n_x4(input_at_3 + 4);
//
//                                float32x4 i01_x4 = concat(i00_x4, i0n_x4, 1);
//                                float32x4 i02_x4 = concat(i00_x4, i0n_x4, 2);
//                                float32x4 i11_x4 = concat(i10_x4, i1n_x4, 1);
//                                float32x4 i12_x4 = concat(i10_x4, i1n_x4, 2);
//                                float32x4 i21_x4 = concat(i20_x4, i2n_x4, 1);
//                                float32x4 i22_x4 = concat(i20_x4, i2n_x4, 2);
//                                float32x4 i31_x4 = concat(i30_x4, i3n_x4, 1);
//                                float32x4 i32_x4 = concat(i30_x4, i3n_x4, 2);
//
//                                float32x4 out00_x4(out_at_0);float32x4 out01_x4(out_at_0 + out_width);
//#else
                            float32x4 i00_x4(input_at_0), i01_x4(input_at_0 + 1), i02_x4(input_at_0 + 2);
                            float32x4 i10_x4(input_at_1), i11_x4(input_at_1 + 1), i12_x4(input_at_1 + 2);
                            float32x4 i20_x4(input_at_2), i21_x4(input_at_2 + 1), i22_x4(input_at_2 + 2);
                            float32x4 i30_x4(input_at_3), i31_x4(input_at_3 + 1), i32_x4(input_at_3 + 2);

                            float32x4 out00_x4(out_at_0), out01_x4(out_at_0 + out_width);
//#endif
                            out00_x4 = fmadd(i00_x4, k00_x4, out00_x4, 0);
                            out00_x4 = fmadd(i01_x4, k00_x4, out00_x4, 1);
                            out00_x4 = fmadd(i02_x4, k00_x4, out00_x4, 2);
                            out00_x4 = fmadd(i10_x4, k01_x4, out00_x4, 0);
                            out00_x4 = fmadd(i11_x4, k01_x4, out00_x4, 1);
                            out00_x4 = fmadd(i12_x4, k01_x4, out00_x4, 2);
                            out00_x4 = fmadd(i20_x4, k02_x4, out00_x4, 0);
                            out00_x4 = fmadd(i21_x4, k02_x4, out00_x4, 1);
                            out00_x4 = fmadd(i22_x4, k02_x4, out00_x4, 2);

                            out01_x4 = fmadd(i10_x4, k00_x4, out01_x4, 0);
                            out01_x4 = fmadd(i11_x4, k00_x4, out01_x4, 1);
                            out01_x4 = fmadd(i12_x4, k00_x4, out01_x4, 2);
                            out01_x4 = fmadd(i20_x4, k01_x4, out01_x4, 0);
                            out01_x4 = fmadd(i21_x4, k01_x4, out01_x4, 1);
                            out01_x4 = fmadd(i22_x4, k01_x4, out01_x4, 2);
                            out01_x4 = fmadd(i30_x4, k02_x4, out01_x4, 0);
                            out01_x4 = fmadd(i31_x4, k02_x4, out01_x4, 1);
                            out01_x4 = fmadd(i32_x4, k02_x4, out01_x4, 2);

                            out00_x4.store(out_at_0);
                            out01_x4.store(out_at_0 + out_width);

                            input_at_0 += 4;
                            input_at_1 += 4;
                            input_at_2 += 4;
                            input_at_3 += 4;

                            out_at_0 += 4;
                        } //out_w
                        input_at_0 += 2 + input_width;
                        input_at_1 += 2 + input_width;
                        input_at_2 += 2 + input_width;
                        input_at_3 += 2 + input_width;

                        out_at_0 += out_width;
                    } //out_h
                } //c
            } //m
        } //n

        if (out_padded_flag) {
            std::array<int, 2> pad_h = {0, src_output_shape[2] - out_height};
            std::array<int, 2> pad_w = {0, src_output_shape[3] - out_width};
            PadAlgorithm<float>::cut2d(out_padded, pad_h, pad_w, 0.f, out);
        }
    }

    template<typename T>
    void Conv2d3x3<T>::conv2d_3x3_s2(const Tensor &x,
                                     const Padding2D &padding,
                                     float padding_value,
                                     const Tensor &w,
                                     Tensor &out) {

    }

    template<>
    void Conv2d3x3<float>::conv2d_3x3_s2(const Tensor &x,
                                         const Padding2D &padding,
                                         float padding_value,
                                         const Tensor &w,
                                         Tensor &out) {
        auto kernel_shape = w.sizes();

        Tensor input_padded = x;
        Tensor out_padded = out;
        bool out_padded_flag = false;
        Stride2D stride(2, 2);
        KSize2D ksize(3, 3);
        KernelCommonFunc<float>::in_out_pad_and_fix_size(x,
                                       kernel_shape,
                                       out,
                                       1,
                                       4,
                                       padding,
                                       padding_value,
                                       stride,
                                       ksize,
                                       input_padded,
                                       out_padded,
                                       out_padded_flag);

        auto output_shape = out_padded.sizes();
        auto input_shape = input_padded.sizes();
        auto src_output_shape = out.sizes();

        const float *kernel_ptr = w.data<float>();
        const float *input_ptr = input_padded.data<float>();
        float *out_ptr = out_padded.data<float>();

        int num = input_shape[0];
        int input_channel = input_shape[1];
        int input_height = input_shape[2];
        int input_width = input_shape[3];
        int out_channel = output_shape[1];
        int out_height = output_shape[2];
        int out_width = output_shape[3];
        int out_channel_offset = out_height * out_width;
        int output_number_offset = out_channel * out_channel_offset;
        int input_channel_offset = input_height * input_width;
        int input_number_offset = input_channel * input_channel_offset;

        for (int n = 0; n < num; ++n) {
            int out_channel_bound = out_channel >> 1;
            int remain_out_channel = out_channel_bound << 1;
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int m = 0; m < out_channel_bound; ++m) {
                int mm = m * 2;
                float *out_at = out_ptr + n * output_number_offset + mm * out_channel_offset;
                float *out_ptr_0 = out_at;
                float *out_ptr_1 = out_ptr_0 + out_channel_offset;
                for (int c = 0; c < input_channel; ++c) {
                    const float *kernel_at = kernel_ptr + mm * input_channel * 9 + c * 9;
                    const float *kernel_at_0 = kernel_at;
                    const float *kernel_at_1 = kernel_at_0 + input_channel * 9;

                    const float *input_at = input_ptr + n * input_number_offset + c * input_channel_offset;
                    const float *input_at_0 = input_at;
                    const float *input_at_1 = input_at_0 + input_width;
                    const float *input_at_2 = input_at_1 + input_width;

                    float *out_at_0 = out_ptr_0;
                    float *out_at_1 = out_ptr_1;

                    float32x4 k00_x4(kernel_at_0), k01_x4(kernel_at_0 + 3), k02_x4(kernel_at_0 + 6);
                    float32x4 k10_x4(kernel_at_1), k11_x4(kernel_at_1 + 3), k12_x4(kernel_at_1 + 6);

                    for (int h = 0; h < out_height; ++h) {
                        for (int w = 0; w < out_width; w += 4) {
                            int in_offset = (h * 2) * input_width + (w * 2);
                            const float *i0 = input_at_0 + in_offset;
                            const float *i1 = input_at_1 + in_offset;
                            const float *i2 = input_at_2 + in_offset;
#ifdef TS_USE_NEON
                            float32x4x2 i00_x4x2 = incx4x2_load(i0, 2);
                            float32x4x2 i0n_x4x2 = incx4x2_load(i0 + 8, 2);
                            float32x4x2 i10_x4x2 = incx4x2_load(i1, 2);
                            float32x4x2 i1n_x4x2 = incx4x2_load(i1 + 8, 2);
                            float32x4x2 i20_x4x2 = incx4x2_load(i2, 2);
                            float32x4x2 i2n_x4x2 = incx4x2_load(i2 + 8, 2);

                            float32x4 i00_x4 = i00_x4x2[0];float32x4 i01_x4 = i00_x4x2[1];float32x4 i02_x4 = concat(i00_x4, i0n_x4x2[0], 1);
                            float32x4 i10_x4 = i10_x4x2[0];float32x4 i11_x4 = i10_x4x2[1];float32x4 i12_x4 = concat(i10_x4, i1n_x4x2[0], 1);
                            float32x4 i20_x4 = i20_x4x2[0];float32x4 i21_x4 = i20_x4x2[1];float32x4 i22_x4 = concat(i20_x4, i2n_x4x2[0], 1);
#else
                            float32x4 i00_x4 = inc_load(i0, 2), i01_x4 = inc_load((i0 + 1), 2), i02_x4 = inc_load(
                                    (i0 + 2), 2);
                            float32x4 i10_x4 = inc_load(i1, 2), i11_x4 = inc_load((i1 + 1), 2), i12_x4 = inc_load(
                                    (i1 + 2), 2);
                            float32x4 i20_x4 = inc_load(i2, 2), i21_x4 = inc_load((i2 + 1), 2), i22_x4 = inc_load(
                                    (i2 + 2), 2);
#endif
                            float32x4 out00_x4(out_at_0);
                            float32x4 out10_x4(out_at_1);

                            out00_x4 = fmadd(i00_x4, k00_x4, out00_x4, 0);
                            out00_x4 = fmadd(i01_x4, k00_x4, out00_x4, 1);
                            out00_x4 = fmadd(i02_x4, k00_x4, out00_x4, 2);
                            out00_x4 = fmadd(i10_x4, k01_x4, out00_x4, 0);
                            out00_x4 = fmadd(i11_x4, k01_x4, out00_x4, 1);
                            out00_x4 = fmadd(i12_x4, k01_x4, out00_x4, 2);
                            out00_x4 = fmadd(i20_x4, k02_x4, out00_x4, 0);
                            out00_x4 = fmadd(i21_x4, k02_x4, out00_x4, 1);
                            out00_x4 = fmadd(i22_x4, k02_x4, out00_x4, 2);

                            out10_x4 = fmadd(i00_x4, k10_x4, out10_x4, 0);
                            out10_x4 = fmadd(i01_x4, k10_x4, out10_x4, 1);
                            out10_x4 = fmadd(i02_x4, k10_x4, out10_x4, 2);
                            out10_x4 = fmadd(i10_x4, k11_x4, out10_x4, 0);
                            out10_x4 = fmadd(i11_x4, k11_x4, out10_x4, 1);
                            out10_x4 = fmadd(i12_x4, k11_x4, out10_x4, 2);
                            out10_x4 = fmadd(i20_x4, k12_x4, out10_x4, 0);
                            out10_x4 = fmadd(i21_x4, k12_x4, out10_x4, 1);
                            out10_x4 = fmadd(i22_x4, k12_x4, out10_x4, 2);

                            out00_x4.store(out_at_0);
                            out10_x4.store(out_at_1);
                            out_at_0 += 4;
                            out_at_1 += 4;
                        } //out_w
                    } //out_h
                } //c
            } //m
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int m = remain_out_channel; m < out_channel; ++m) {
                float *out_at = out_ptr + n * output_number_offset + m * out_channel_offset;
                float *out_ptr_0 = out_at;

                for (int c = 0; c < input_channel; ++c) {
                    const float *kernel_at = kernel_ptr + m * input_channel * 9 + c * 9;
                    const float *kernel_at_0 = kernel_at;

                    const float *input_at = input_ptr + n * input_number_offset + c * input_channel_offset;
                    const float *input_at_0 = input_at;
                    const float *input_at_1 = input_at_0 + input_width;
                    const float *input_at_2 = input_at_1 + input_width;

                    float *out_at_0 = out_ptr_0;

                    float32x4 k00_x4(kernel_at_0), k01_x4(kernel_at_0 + 3), k02_x4(kernel_at_0 + 6);

                    for (int h = 0; h < out_height; ++h) {
                        for (int w = 0; w + 3 < out_width; w += 4) {
                            int in_offset = (h * 2) * input_width + (w * 2);
                            const float *i0 = input_at_0 + in_offset;
                            const float *i1 = input_at_1 + in_offset;
                            const float *i2 = input_at_2 + in_offset;
#ifdef TS_USE_NEON
                            float32x4x2 i00_x4x2 = incx4x2_load(i0, 2);
                            float32x4x2 i0n_x4x2 = incx4x2_load(i0 + 8, 2);
                            float32x4x2 i10_x4x2 = incx4x2_load(i1, 2);
                            float32x4x2 i1n_x4x2 = incx4x2_load(i1 + 8, 2);
                            float32x4x2 i20_x4x2 = incx4x2_load(i2, 2);
                            float32x4x2 i2n_x4x2 = incx4x2_load(i2 + 8, 2);

                            float32x4 i00_x4 = i00_x4x2[0];float32x4 i01_x4 = i00_x4x2[1];float32x4 i02_x4 = concat(i00_x4, i0n_x4x2[0], 1);
                            float32x4 i10_x4 = i10_x4x2[0];float32x4 i11_x4 = i10_x4x2[1];float32x4 i12_x4 = concat(i10_x4, i1n_x4x2[0], 1);
                            float32x4 i20_x4 = i20_x4x2[0];float32x4 i21_x4 = i20_x4x2[1];float32x4 i22_x4 = concat(i20_x4, i2n_x4x2[0], 1);
#else
                            float32x4 i00_x4 = inc_load(i0, 2), i01_x4 = inc_load((i0 + 1), 2), i02_x4 = inc_load(
                                    (i0 + 2), 2);
                            float32x4 i10_x4 = inc_load(i1, 2), i11_x4 = inc_load((i1 + 1), 2), i12_x4 = inc_load(
                                    (i1 + 2), 2);
                            float32x4 i20_x4 = inc_load(i2, 2), i21_x4 = inc_load((i2 + 1), 2), i22_x4 = inc_load(
                                    (i2 + 2), 2);
#endif
                            float32x4 out00_x4(out_at_0);

                            out00_x4 = fmadd(i00_x4, k00_x4, out00_x4, 0);
                            out00_x4 = fmadd(i01_x4, k00_x4, out00_x4, 1);
                            out00_x4 = fmadd(i02_x4, k00_x4, out00_x4, 2);
                            out00_x4 = fmadd(i10_x4, k01_x4, out00_x4, 0);
                            out00_x4 = fmadd(i11_x4, k01_x4, out00_x4, 1);
                            out00_x4 = fmadd(i12_x4, k01_x4, out00_x4, 2);
                            out00_x4 = fmadd(i20_x4, k02_x4, out00_x4, 0);
                            out00_x4 = fmadd(i21_x4, k02_x4, out00_x4, 1);
                            out00_x4 = fmadd(i22_x4, k02_x4, out00_x4, 2);

                            out00_x4.store(out_at_0);
                            out_at_0 += 4;
                        }
                    }
                }
            }
        } //n

        if (out_padded_flag) {
            std::array<int, 2> pad_h = {0, src_output_shape[2] - out_height};
            std::array<int, 2> pad_w = {0, src_output_shape[3] - out_width};
            PadAlgorithm<float>::cut2d(out_padded, pad_h, pad_w, 0.f, out);
        }
    }
}//arm
}//cpu
}//ts
