//
// Created by yang on 2019/11/19.
//

#include "kernels/cpu/arm/conv2d_5x5.h"
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

    template <typename T>
    void Conv2d5x5<T>::conv2d_5x5_s1(const Tensor &x,
                              const Padding2D &padding,
                              float padding_value,
                              const Tensor &w,
                              Tensor &out){

    }

    //TODO:output:2channel*2height*4width to optimize cache
    // 4channel*1height*4width rather than 2c2h4w due to i don't know how to optimize pipeline
    // when i use neon instruction
    template <>
    void Conv2d5x5<float>::conv2d_5x5_s1(const Tensor &x,
                                     const Padding2D &padding,
                                     float padding_value,
                                     const Tensor &w,
                                     Tensor &out){
        auto kernel_shape = w.sizes();

        Tensor input_padded = x;
        Tensor out_padded = out;
        bool out_padded_flag = false;
        Stride2D stride(1, 1);
        KSize2D ksize(5, 5);
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
            int out_channel_bound = out_channel >> 2;
            int remain_out_channel = out_channel_bound << 2;
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int m = 0; m < out_channel_bound; ++m) {
                int mm = m * 4;
                float *out_at = out_ptr + n * output_number_offset + mm * out_channel_offset;
                float *out_ptr_0 = out_at;
                float *out_ptr_1 = out_ptr_0 + out_channel_offset;
                float *out_ptr_2 = out_ptr_1 + out_channel_offset;
                float *out_ptr_3 = out_ptr_2 + out_channel_offset;

                for (int c = 0; c < input_channel; ++c) {
                    const float *kernel_at = kernel_ptr + mm * input_channel * 25 + c * 25;
                    const float *kernel_at_0 = kernel_at;
                    const float *kernel_at_1 = kernel_at_0 + input_channel * 25;
                    const float *kernel_at_2 = kernel_at_1 + input_channel * 25;
                    const float *kernel_at_3 = kernel_at_2 + input_channel * 25;

                    const float *input_at = input_ptr + n * input_number_offset + c * input_channel_offset;
                    const float *input_at_0 = input_at;

                    float *out_at_0 = out_ptr_0;float *out_at_1 = out_ptr_1;
                    float *out_at_2 = out_ptr_2;float *out_at_3 = out_ptr_3;

                    for (int h = 0; h < out_height; ++h) {
                        for (int w = 0; w + 3 < out_width; w += 4) {
                            float32x4 out00_x4(out_at_0);
                            float32x4 out10_x4(out_at_1);
                            float32x4 out20_x4(out_at_2);
                            float32x4 out30_x4(out_at_3);

                            //for input line 5
                            const float* kernel_cur_0 = kernel_at_0;
                            const float* kernel_cur_1 = kernel_at_1;
                            const float* kernel_cur_2 = kernel_at_2;
                            const float* kernel_cur_3 = kernel_at_3;

                            const float* input_cur_0 = input_at_0;

                            for (int i = 0; i < 5; ++i) {
                                float32x4 i00_x4(input_cur_0);
                                float32x4 i04_x4(input_cur_0 + 4);
                                float32x4 i01_x4 = concat(i00_x4, i04_x4, 1);
                                float32x4 i02_x4 = concat(i00_x4, i04_x4, 2);
                                float32x4 i03_x4 = concat(i00_x4, i04_x4, 3);

                                float32x4 k00_x4(kernel_cur_0), k00_x4_sup(kernel_cur_0 + 3);
                                float32x4 k10_x4(kernel_cur_1), k10_x4_sup(kernel_cur_1 + 3);
                                float32x4 k20_x4(kernel_cur_2), k20_x4_sup(kernel_cur_2 + 3);
                                float32x4 k30_x4(kernel_cur_3), k30_x4_sup(kernel_cur_3 + 3);

                                out00_x4 = fmadd(i00_x4, k00_x4, out00_x4, 0);
                                out00_x4 = fmadd(i01_x4, k00_x4, out00_x4, 1);
                                out00_x4 = fmadd(i02_x4, k00_x4, out00_x4, 2);
                                out00_x4 = fmadd(i03_x4, k00_x4, out00_x4, 3);
                                out00_x4 = fmadd(i04_x4, k00_x4_sup, out00_x4, 1);

                                out10_x4 = fmadd(i00_x4, k10_x4, out10_x4, 0);
                                out10_x4 = fmadd(i01_x4, k10_x4, out10_x4, 1);
                                out10_x4 = fmadd(i02_x4, k10_x4, out10_x4, 2);
                                out10_x4 = fmadd(i03_x4, k10_x4, out10_x4, 3);
                                out10_x4 = fmadd(i04_x4, k10_x4_sup, out10_x4, 1);

                                out20_x4 = fmadd(i00_x4, k20_x4, out20_x4, 0);
                                out20_x4 = fmadd(i01_x4, k20_x4, out20_x4, 1);
                                out20_x4 = fmadd(i02_x4, k20_x4, out20_x4, 2);
                                out20_x4 = fmadd(i03_x4, k20_x4, out20_x4, 3);
                                out20_x4 = fmadd(i04_x4, k20_x4_sup, out20_x4, 1);

                                out30_x4 = fmadd(i00_x4, k30_x4, out30_x4, 0);
                                out30_x4 = fmadd(i01_x4, k30_x4, out30_x4, 1);
                                out30_x4 = fmadd(i02_x4, k30_x4, out30_x4, 2);
                                out30_x4 = fmadd(i03_x4, k30_x4, out30_x4, 3);
                                out30_x4 = fmadd(i04_x4, k30_x4_sup, out30_x4, 1);

                                kernel_cur_0 += 5;
                                kernel_cur_1 += 5;
                                kernel_cur_2 += 5;
                                kernel_cur_3 += 5;

                                input_cur_0 += input_width;
                            }  //i
                            out00_x4.store(out_at_0);
                            out10_x4.store(out_at_1);
                            out20_x4.store(out_at_2);
                            out30_x4.store(out_at_3);

                            input_at_0 += 4;
                            out_at_0 += 4;
                            out_at_1 += 4;
                            out_at_2 += 4;
                            out_at_3 += 4;
                        } //w
                        input_at_0 += 4;
                    } //h
                } //c
            } //m
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int m = remain_out_channel; m < out_channel; ++m) {
                float *out_at = out_ptr + n * output_number_offset + m * out_channel_offset;
                float *out_ptr_0 = out_at;

                for (int c = 0; c < input_channel; ++c) {
                    const float *kernel_at = kernel_ptr + m * input_channel * 25 + c * 25;
                    const float *kernel_at_0 = kernel_at;

                    const float *input_at = input_ptr + n * input_number_offset + c * input_channel_offset;
                    const float *input_at_0 = input_at;

                    float *out_at_0 = out_ptr_0;

                    for (int h = 0; h < out_height; ++h) {
                        for (int w = 0; w + 3 < out_width; w += 4) {
                            float32x4 out00_x4(out_at_0);

                            const float* kernel_cur_0 = kernel_at_0;
                            const float* input_cur_0 = input_at_0;

                            //for input line
                            for (int i = 0; i < 5; ++i) {
                                float32x4 i00_x4(input_cur_0);
                                float32x4 i04_x4(input_cur_0 + 4);
                                float32x4 i01_x4 = concat(i00_x4, i04_x4, 1);
                                float32x4 i02_x4 = concat(i00_x4, i04_x4, 2);
                                float32x4 i03_x4 = concat(i00_x4, i04_x4, 3);

                                float32x4 k00_x4(kernel_cur_0), k00_x4_sup(kernel_cur_0 + 3);

                                out00_x4 = fmadd(i00_x4, k00_x4, out00_x4, 0);
                                out00_x4 = fmadd(i01_x4, k00_x4, out00_x4, 1);
                                out00_x4 = fmadd(i02_x4, k00_x4, out00_x4, 2);
                                out00_x4 = fmadd(i03_x4, k00_x4, out00_x4, 3);
                                out00_x4 = fmadd(i04_x4, k00_x4_sup, out00_x4, 1);

                                kernel_cur_0 += 5;
                                input_cur_0 += input_width;
                            } //i
                            out00_x4.store(out_at_0);
                            input_at_0 += 4;
                            out_at_0 += 4;
                        } //w
                        input_at_0 += 4;
                    } //h
                } //c
            } //m
        } //n

        if (out_padded_flag) {
            std::array<int, 2> pad_h = {0, src_output_shape[2] - out_height};
            std::array<int, 2> pad_w = {0, src_output_shape[3] - out_width};
            PadAlgorithm<float>::cut2d(out_padded, pad_h, pad_w, 0.f, out);
        }
    }

    template <typename T>
    void Conv2d5x5<T>::conv2d_5x5_s2(const Tensor &x,
                                     const Padding2D &padding,
                                     float padding_value,
                                     const Tensor &w,
                                     Tensor &out){

    }

    template <>
    void Conv2d5x5<float>::conv2d_5x5_s2(const Tensor &x,
                                         const Padding2D &padding,
                                         float padding_value,
                                         const Tensor &w,
                                         Tensor &out) {
        auto kernel_shape = w.sizes();

        Tensor input_padded = x;
        Tensor out_padded = out;
        bool out_padded_flag = false;
        Stride2D stride(2, 2);
        KSize2D ksize(5, 5);
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
            int out_channel_bound = out_channel >> 2;
            int remain_out_channel = out_channel_bound << 2;
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int m = 0; m < out_channel_bound; ++m) {
                int mm = m * 4;
                float *out_at = out_ptr + n * output_number_offset + mm * out_channel_offset;
                float *out_ptr_0 = out_at;
                float *out_ptr_1 = out_ptr_0 + out_channel_offset;
                float *out_ptr_2 = out_ptr_1 + out_channel_offset;
                float *out_ptr_3 = out_ptr_2 + out_channel_offset;

                for (int c = 0; c < input_channel; ++c) {
                    const float *kernel_at = kernel_ptr + mm * input_channel * 25 + c * 25;
                    const float *kernel_at_0 = kernel_at;
                    const float *kernel_at_1 = kernel_at_0 + input_channel * 25;
                    const float *kernel_at_2 = kernel_at_1 + input_channel * 25;
                    const float *kernel_at_3 = kernel_at_2 + input_channel * 25;

                    const float *input_at = input_ptr + n * input_number_offset + c * input_channel_offset;
                    const float *input_at_0 = input_at;

                    float *out_at_0 = out_ptr_0;
                    float *out_at_1 = out_ptr_1;
                    float *out_at_2 = out_ptr_2;
                    float *out_at_3 = out_ptr_3;

                    for (int h = 0; h < out_height; ++h) {
                        for (int w = 0; w + 3 < out_width; w += 4) {
                            int in_offset = h * 2 * input_width + w * 2;

                            float32x4 out00_x4(out_at_0);
                            float32x4 out10_x4(out_at_1);
                            float32x4 out20_x4(out_at_2);
                            float32x4 out30_x4(out_at_3);

                            const float *kernel_cur_0 = kernel_at_0;
                            const float *kernel_cur_1 = kernel_at_1;
                            const float *kernel_cur_2 = kernel_at_2;
                            const float *kernel_cur_3 = kernel_at_3;

                            const float *input_cur_0 = input_at_0 + in_offset;

                            for (int i = 0; i < 5; ++i) {
                                float32x4x2 i0246_1357 = incx4x2_load(input_cur_0, 2);
                                float32x4x2 i0246_1357_sup = incx4x2_load(input_cur_0 + 8, 2);
//                                float32x4 i00_x4 = i0246_1357[0];
//                                float32x4 i01_x4 = i0246_1357[1];
                                float32x4 i02_x4 = concat(i0246_1357[0], i0246_1357_sup[0], 1);
                                float32x4 i03_x4 = concat(i0246_1357[1], i0246_1357_sup[1], 1);
                                float32x4 i04_x4 = concat(i0246_1357[0], i0246_1357_sup[0], 2);

                                float32x4 k00_x4(kernel_cur_0), k00_x4_sup(kernel_cur_0 + 3);
                                float32x4 k10_x4(kernel_cur_1), k10_x4_sup(kernel_cur_1 + 3);
                                float32x4 k20_x4(kernel_cur_2), k20_x4_sup(kernel_cur_2 + 3);
                                float32x4 k30_x4(kernel_cur_3), k30_x4_sup(kernel_cur_3 + 3);

                                out00_x4 = fmadd(i0246_1357[0], k00_x4, out00_x4, 0);
                                out00_x4 = fmadd(i0246_1357[1], k00_x4, out00_x4, 1);
                                out00_x4 = fmadd(i02_x4, k00_x4, out00_x4, 2);
                                out00_x4 = fmadd(i03_x4, k00_x4, out00_x4, 3);
                                out00_x4 = fmadd(i04_x4, k00_x4_sup, out00_x4, 1);

                                out10_x4 = fmadd(i0246_1357[0], k10_x4, out10_x4, 0);
                                out10_x4 = fmadd(i0246_1357[1], k10_x4, out10_x4, 1);
                                out10_x4 = fmadd(i02_x4, k10_x4, out10_x4, 2);
                                out10_x4 = fmadd(i03_x4, k10_x4, out10_x4, 3);
                                out10_x4 = fmadd(i04_x4, k10_x4_sup, out10_x4, 1);

                                out20_x4 = fmadd(i0246_1357[0], k20_x4, out20_x4, 0);
                                out20_x4 = fmadd(i0246_1357[1], k20_x4, out20_x4, 1);
                                out20_x4 = fmadd(i02_x4, k20_x4, out20_x4, 2);
                                out20_x4 = fmadd(i03_x4, k20_x4, out20_x4, 3);
                                out20_x4 = fmadd(i04_x4, k20_x4_sup, out20_x4, 1);

                                out30_x4 = fmadd(i0246_1357[0], k30_x4, out30_x4, 0);
                                out30_x4 = fmadd(i0246_1357[1], k30_x4, out30_x4, 1);
                                out30_x4 = fmadd(i02_x4, k30_x4, out30_x4, 2);
                                out30_x4 = fmadd(i03_x4, k30_x4, out30_x4, 3);
                                out30_x4 = fmadd(i04_x4, k30_x4_sup, out30_x4, 1);

                                kernel_cur_0 += 5;
                                kernel_cur_1 += 5;
                                kernel_cur_2 += 5;
                                kernel_cur_3 += 5;

                                input_cur_0 += input_width;
                            } //i
                            out00_x4.store(out_at_0);
                            out10_x4.store(out_at_1);
                            out20_x4.store(out_at_2);
                            out30_x4.store(out_at_3);

                            out_at_0 += 4;
                            out_at_1 += 4;
                            out_at_2 += 4;
                            out_at_3 += 4;
                        } //w
                    } //h
                } //c
            } //m
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int m = remain_out_channel; m < out_channel; ++m) {
                float *out_at = out_ptr + n * output_number_offset + m * out_channel_offset;
                float *out_ptr_0 = out_at;

                for (int c = 0; c < input_channel; ++c) {
                    const float *kernel_at = kernel_ptr + m * input_channel * 25 + c * 25;
                    const float *kernel_at_0 = kernel_at;

                    const float *input_at = input_ptr + n * input_number_offset + c * input_channel_offset;
                    const float *input_at_0 = input_at;

                    float *out_at_0 = out_ptr_0;

                    for (int h = 0; h < out_height; ++h) {
                        for (int w = 0; w + 3 < out_width; w += 4) {
                            int in_offset = h * 2 * input_width + w * 2;

                            float32x4 out00_x4(out_at_0);

                            const float *kernel_cur_0 = kernel_at_0;
                            const float *input_cur_0 = input_at_0 + in_offset;

                            for (int i = 0; i < 5; ++i) {
                                float32x4x2 i0246_1357 = incx4x2_load(input_cur_0, 2);
                                float32x4x2 i0246_1357_sup = incx4x2_load(input_cur_0 + 8, 2);
//                                float32x4 i00_x4 = i0246_1357[0];
//                                float32x4 i01_x4 = i0246_1357[1];
                                float32x4 i02_x4 = concat(i0246_1357[0], i0246_1357_sup[0], 1);
                                float32x4 i03_x4 = concat(i0246_1357[1], i0246_1357_sup[1], 1);
                                float32x4 i04_x4 = concat(i0246_1357[0], i0246_1357_sup[0], 2);

                                float32x4 k00_x4(kernel_cur_0), k00_x4_sup(kernel_cur_0 + 3);

                                out00_x4 = fmadd(i0246_1357[0], k00_x4, out00_x4, 0);
                                out00_x4 = fmadd(i0246_1357[1], k00_x4, out00_x4, 1);
                                out00_x4 = fmadd(i02_x4, k00_x4, out00_x4, 2);
                                out00_x4 = fmadd(i03_x4, k00_x4, out00_x4, 3);
                                out00_x4 = fmadd(i04_x4, k00_x4_sup, out00_x4, 1);

                                kernel_cur_0 += 5;
                                input_cur_0 += input_width;
                            } //i
                            out00_x4.store(out_at_0);
                            out_at_0 += 4;
                        } //w
                    } //h
                } //c
            } //m
        } //n
        if (out_padded_flag) {
            std::array<int, 2> pad_h = {0, src_output_shape[2] - out_height};
            std::array<int, 2> pad_w = {0, src_output_shape[3] - out_width};
            PadAlgorithm<float>::cut2d(out_padded, pad_h, pad_w, 0.f, out);
        }
    }
} //arm
} //cpu
} //ts




