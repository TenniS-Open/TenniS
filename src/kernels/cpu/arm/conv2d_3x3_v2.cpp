//
// Created by yang on 2019/12/16.
//

#include "kernels/cpu/arm/conv2d_3x3_v2.h"
#include "kernels/common/simd_def/simd_neon_def.h"
#include "kernels/cpu/pad2d_algorithm.h"
#include "kernels/common/function.h"
#ifdef TS_USE_OPENMP
#include "kernels/common/openmp.h"
#endif

#include <array>


namespace ts {
namespace cpu {
namespace arm {

     template<typename T>
     void Conv2d3x3V2<T>::conv2d_3x3_s1(const Tensor &x,
                                        const Padding2D &padding,
                                        float padding_value,
                                        const Tensor &w,
                                        Tensor &out) {

     }

     //TODO:add armv7 and pure c support
     template<>
     void Conv2d3x3V2<float>::conv2d_3x3_s1(const Tensor &x,
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
                     float *out_at_0n = out_at_0 + out_width;
                     float *out_at_1 = out_ptr_1;
                     float *out_at_1n = out_at_1 + out_width;

                    //float32x4 k00_x4(kernel_at_0), k01_x4(kernel_at_0 + 3), k02_x4(kernel_at_0 + 6);
                    //float32x4 k10_x4(kernel_at_1), k11_x4(kernel_at_1 + 3), k12_x4(kernel_at_1 + 6);
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
                     float32x4_t k00_x4 = vld1q_f32(kernel_at_0);
                     float32x4_t k01_x4 = vld1q_f32(kernel_at_0 + 3);
                     float32x4_t k02_x4 = vld1q_f32(kernel_at_0 + 6);
                     float32x4_t k10_x4 = vld1q_f32(kernel_at_1);
                     float32x4_t k11_x4 = vld1q_f32(kernel_at_1 + 3);
                     float32x4_t k12_x4 = vld1q_f32(kernel_at_1 + 6);
#endif

                     for (int h = 0; h + 1 < out_height; h += 2) {
                         int w_count = out_width >> 2;
                         if (w_count > 0) {
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#if __aarch64__
                             asm volatile(
                             "prfm   pldl1keep, [%5, #256]         \n"
                             "ld1    {v8.4s, v9.4s}, [%5]          \n" //input_at_0->v8,v9
                             "add    %5, %5, #16                   \n" //input_at_0 += 4

                             "prfm   pldl1keep, [%8, #256]         \n"
                             "ld1    {v12.4s, v13.4s}, [%8]        \n" //input_at_3->v12,v13
                             "add    %8, %8, #16                   \n" //input_at_3 += 4

                             "ext    v10.16b, v8.16b, v9.16b, #4   \n" //v8.4s[0,1,2,3],v9.4s[a,b,c,d]->v10.4s[1,2,3,a]
                             "ext    v11.16b, v12.16b, v13.16b, #4 \n" //v12.4s[0,1,2,3],v13.4s[a,b,c,d]->v11.4s[1,2,3,a]

                             "0:                                   \n" //loop w_count

                             "prfm   pldl1keep, [%1, #128]         \n"
                             "ld1    {v4.4s}, [%1]                 \n" //output_at_0->v4.4s

                             "prfm   pldl1keep, [%2, #128]         \n"
                             "ld1    {v5.4s}, [%2]                 \n" //output_at_1->v5.4s

                             "prfm   pldl1keep, [%3, #128]         \n"
                             "ld1    {v6.4s}, [%3]                 \n" //output_at_0n->v6.4s

                             "prfm   pldl1keep, [%4, #128]         \n"
                             "ld1    {v7.4s}, [%4]                 \n" //output_at_1n->v7.4s

                             "fmla   v4.4s, v8.4s, %18.s[0]        \n" //out_0_x4 += i00 * k00.0
                             "fmla   v5.4s, v8.4s, %21.s[0]        \n" //out_1_x4 += i00 * k10.0
                             "fmla   v6.4s, v12.4s, %20.s[0]       \n" //out_0n_x4 += i30 * k02.0
                             "fmla   v7.4s, v12.4s, %23.s[0]       \n" //out_1n_x4 += i30 * k12.0

                             "ext    v8.16b, v8.16b, v9.16b, #8    \n" //v8.4s[0,1,2,3],v9[a,b,c,d]->v8.4s[2,3,a,b]
                             "ext    v9.16b, v12.16b, v13.16b, #8  \n" //v12.4s[0,1,2,3],v13[a,b,c,d]->v9.4s[2,3,a,b]

                             "fmla   v4.4s, v10.4s, %18.s[1]       \n" //out_0_x4 += i01 * k00.1
                             "fmla   v5.4s, v10.4s, %21.s[1]       \n" //out_1_x4 += i01 * k10.1
                             "fmla   v6.4s, v11.4s, %20.s[1]       \n" //out_0n_x4 += i31 * k02.1
                             "fmla   v7.4s, v11.4s, %23.s[1]       \n" //out_1n_x4 += i31 * k12.1

                             "prfm   pldl1keep, [%6, #256]         \n"
                             "ld1    {v12.4s, v13.4s}, [%6]        \n" //input_at_1->v12.4s,v13.4s
                             "add    %6, %6, #16                   \n" //input_at_1 += 4

                             "fmla   v4.4s, v8.4s, %18.s[2]        \n" //out_0_x4 += i02 * k00.2
                             "fmla   v5.4s, v8.4s, %21.s[2]        \n" //out_1_x4 += i02 * k10.2
                             "fmla   v6.4s, v9.4s, %20.s[2]        \n" //out_0n_x4 += i32 * k02.2
                             "fmla   v7.4s, v9.4s, %23.s[2]        \n" //out_1n_x4 += i32 * k12.2

                             "ext    v10.16b, v12.16b, v13.16b, #4 \n" //v12.4s[0,1,2,3],v13.4s[a,b,c,d]->v10.4s[1,2,3,a]

                             "fmla   v4.4s, v12.4s, %19.s[0]       \n" //out_0_x4 += i10 * k01.0
                             "fmla   v5.4s, v12.4s, %22.s[0]       \n" //out_1_x4 += i10 * k11.0
                             "fmla   v6.4s, v12.4s, %18.s[0]       \n" //out_0n_x4 += i10 * k00.0
                             "fmla   v7.4s, v12.4s, %21.s[0]       \n" //out_1n_x4 += i10 * k10.0

                             "ext    v11.16b, v12.16b, v13.16b, #8 \n" //v12.4s[0,1,2,3],v13.4s[a,b,c,d]->v11.4s[2,3,a,b]

                             "fmla   v4.4s, v10.4s, %19.s[1]       \n" //out_0_x4 += i11 * k01.1
                             "fmla   v5.4s, v10.4s, %22.s[1]       \n" //out_1_x4 += i11 * k11.1
                             "fmla   v6.4s, v10.4s, %18.s[1]       \n" //out_0n_x4 += i11 * k00.1
                             "fmla   v7.4s, v10.4s, %21.s[1]       \n" //out_1n_x4 += i11 * k10.1

                             "prfm   pldl1keep, [%7, #256]         \n"
                             "ld1    {v8.4s, v9.4s}, [%7]          \n" //input_at_2->v8.4s,v9.4s
                             "add    %7, %7, #16                   \n" //input_at_2 += 4;

                             "fmla   v4.4s, v11.4s, %19.s[2]       \n" //out_0_x4 += i12 * k01.2
                             "fmla   v5.4s, v11.4s, %22.s[2]       \n" //out_1_x4 += i12 * k11.2
                             "fmla   v6.4s, v11.4s, %18.s[2]       \n" //out_0n_x4 += i12 * k00.2
                             "fmla   v7.4s, v11.4s, %21.s[2]       \n" //out_1n_x4 += i12 * k10.2

                             "ext    v10.16b, v8.16b, v9.16b, #4   \n" //v8.4s[0,1,2,3],v9.4s[a,b,c,d]->v10.4s[1,2,3,a]

                             "fmla   v4.4s, v8.4s, %20.s[0]        \n" //out_0_x4 += i20 * k02.0
                             "fmla   v5.4s, v8.4s, %23.s[0]        \n" //out_1_x4 += i20 * k12.0
                             "fmla   v6.4s, v8.4s, %19.s[0]        \n" //out_0n_x4 += i20 * k01.0
                             "fmla   v7.4s, v8.4s, %22.s[0]        \n" //out_1n_x4 += i20 * k11.0

                             "ext    v11.16b, v8.16b, v9.16b, #8   \n" //v8.4s[0,1,2,3],v9.4s[a,b,c,d]->v11.4s[2,3,a,b]

                             "fmla   v4.4s, v10.4s, %20.s[1]       \n" //out_0_x4 += i21 * k02.1
                             "fmla   v5.4s, v10.4s, %23.s[1]       \n" //out_1_x4 += i21 * k12.1
                             "fmla   v6.4s, v10.4s, %19.s[1]       \n" //out_0n_x4 += i21 * k01.1
                             "fmla   v7.4s, v10.4s, %22.s[1]       \n" //out_1n_x4 += i21 * k11.1

                             "prfm   pldl1keep, [%5, #256]         \n"
                             "ld1    {v8.4s, v9.4s}, [%5]          \n" //input_at_0
                             "add    %5, %5, #16                   \n" //input_at_0 += 4

                             "fmla   v4.4s, v11.4s, %20.s[2]       \n" //out_0_x4 += i22 * k02.2
                             "fmla   v5.4s, v11.4s, %23.s[2]       \n" //out_1_x4 += i22 * k12.2
                             "fmla   v6.4s, v11.4s, %19.s[2]       \n" //out_0n_x4 += i22 * k01.2
                             "fmla   v7.4s, v11.4s, %22.s[2]       \n" //out_1n_x4 += i22 * k11.2

                             "prfm   pldl1keep, [%8, #256]         \n"
                             "ld1    {v12.4s, v13.4s}, [%8]        \n" //input_at_3
                             "add    %8, %8, #16                   \n" //input_at_3 += 4

                             "ext    v10.16b, v8.16b, v9.16b, #4   \n" //v8.4s[0,1,2,3],v9.4s[a,b,c,d]->v10.4s[1,2,3,a]

                             "st1    {v4.4s}, [%1], #16            \n" //save output_at_0
                             "st1    {v5.4s}, [%2], #16            \n" //save output_at_1

                             "ext    v11.16b, v12.16b, v13.16b, #4 \n" //v12.4s[0,1,2,3],v13.4s[a,b,c,d]->v11.4s[1,2,3,a]

                             "st1    {v6.4s}, [%3], #16            \n" //save output_at_0n
                             "st1    {v7.4s}, [%4], #16            \n" //save output_at_1n

                             "subs   %w0, %w0, #1                  \n" //w_count -= 1
                             "bne    0b                            \n"

                             "sub    %5, %5, #16                   \n" //input_at_0 -= 4
                             "sub    %8, %8, #16                   \n" //input_at_3 -= 4
                             : "=r"(w_count),                          //%0
                               "=r"(out_at_0),                         //%1
                               "=r"(out_at_1),                         //%2
                               "=r"(out_at_0n),                        //%3
                               "=r"(out_at_1n),                        //%4
                               "=r"(input_at_0),                       //%5
                               "=r"(input_at_1),                       //%6
                               "=r"(input_at_2),                       //%7
                               "=r"(input_at_3)                        //%8
                             : "0"(w_count),
                               "1"(out_at_0),
                               "2"(out_at_1),
                               "3"(out_at_0n),
                               "4"(out_at_1n),
                               "5"(input_at_0),
                               "6"(input_at_1),
                               "7"(input_at_2),
                               "8"(input_at_3),
                               "w"(k00_x4),                      //%18
                               "w"(k01_x4),                      //%19
                               "w"(k02_x4),                      //%20
                               "w"(k10_x4),                      //%21
                               "w"(k11_x4),                      //%22
                               "w"(k12_x4)                       //%23
                             : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13"
                             );
#endif //__aarch64__
#endif //__ARM_NEON__
                         }
                         input_at_0 += 2 + input_width;
                         input_at_1 += 2 + input_width;
                         input_at_2 += 2 + input_width;
                         input_at_3 += 2 + input_width;

                         out_at_0 += out_width;out_at_0n += out_width;
                         out_at_1 += out_width;out_at_1n += out_width;
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
                       const float *kernel_at = kernel_ptr + m * input_channel * 9 + c * 9;
                       const float *kernel_at_0 = kernel_at;

                       const float *input_at = input_ptr + n * input_number_offset + c * input_channel_offset;
                       const float *input_at_0 = input_at;
                       const float *input_at_1 = input_at_0 + input_width;
                       const float *input_at_2 = input_at_1 + input_width;
                       const float *input_at_3 = input_at_2 + input_width;

                       float *out_at_0 = out_ptr_0;
                       float *out_at_0n = out_at_0 + out_width;

                       //float32x4 k00_x4(kernel_at_0), k01_x4(kernel_at_0 + 3), k02_x4(kernel_at_0 + 6);
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
                       float32x4_t k00_x4 = vld1q_f32(kernel_at_0);
                       float32x4_t k01_x4 = vld1q_f32(kernel_at_0 + 3);
                       float32x4_t k02_x4 = vld1q_f32(kernel_at_0 + 6);
#endif

                       for (int h = 0; h + 1 < out_height; h += 2) {
                           int w_count = out_width >> 2;
                           if (w_count > 0) {
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#if __aarch64__
                               asm volatile(
                               "prfm   pldl1keep, [%3, #256]         \n"
                               "ld1    {v9.4s, v10.4s}, [%3]         \n" //input_at_0->v9.4s,v10.4s
                               "add    %3, %3, #16                   \n"

                               "ext    v11.16b, v9.16b, v10.16b, #4  \n" //v9.4s[0,1,2,3],v10.4s[a,b,c,d]->v11.4s[1,2,3,a]
                               "ext    v12.16b, v9.16b, v10.16b, #8  \n" //v9.4s[0,1,2,3],v10.4s[a,b,c,d]->v12.4s[2,3,a,b]

                               "0:                                   \n"

                               "prfm   pldl1keep, [%1, #128]         \n"
                               "ld1    {v7.4s}, [%1]                 \n" //output_at_0->v7.4s

                               "fmla   v7.4s, v9.4s, %14.s[0]        \n" //output_at_0 += i00 * k00.0
                               "fmul   v6.4s, v11.4s, %14.s[1]       \n" //i01 * k00.1
                               "fmul   v13.4s, v12.4s, %14.s[2]      \n" //i02 * k00.2

                               "prfm   pldl1keep, [%4, #256]         \n"
                               "ld1    {v9.4s, v10.4s}, [%4]         \n" //input_at_1->v9.4s, v10.4s
                               "add    %4, %4, #16                   \n"

                               "fmla   v7.4s, v9.4s, %15.s[0]        \n" //output_at_0 += i10 * k01.0

                               "ext    v11.16b, v9.16b, v10.16b, #4  \n" //v9.4s[0,1,2,3],v10.4s[a,b,c,d]->v11.4s[1,2,3,a]
                               "ext    v12.16b, v9.16b, v10.16b, #8  \n" //v9.4s[0,1,2,3],v10.4s[a,b,c,d]->v12.4s[2,3,a,b]

                               "fmla   v6.4s, v11.4s, %15.s[1]       \n" //i11 * k01.1
                               "fmla   v13.4s, v12.4s, %15.s[2]      \n" //i12 * k01.2

                               "prfm   pldl1keep, [%2, #128]         \n"
                               "ld1    {v8.4s}, [%2]                 \n" //output_at_0n->v8.4s

                               "fmla   v8.4s, v9.4s, %14.s[0]        \n" //output_at_0n += i10 * k00.0
                               "fmul   v14.4s, v11.4s, %14.s[1]      \n" //i11 * k00.1
                               "fmul   v15.4s, v12.4s, %14.s[2]      \n" //i12 * k00.2

                               "prfm   pldl1keep, [%5, #256]         \n"
                               "ld1    {v9.4s, v10.4s}, [%5]         \n" //input_at_2->v9.4s,v10.4s
                               "add    %5, %5, #16                   \n"

                               "ext    v11.16b, v9.16b, v10.16b, #4  \n" //v9.4s[0,1,2,3],v10.4s[a,b,c,d]->v11[1,2,3,a]
                               "ext    v12.16b, v9.16b, v10.16b, #8  \n" //v9.4s[0,1,2,3],v10.4s[a,b,c,d]->v11[2,3,a,b]

                               "fmla   v7.4s, v9.4s, %16.s[0]        \n" //out_at_0 += i20 * k02.0
                               "fmla   v6.4s, v11.4s, %16.s[1]       \n" //i21 * k02.1
                               "fmla   v13.4s, v12.4s, %16.s[2]      \n" //i22 * k02.2

                               "fmla   v8.4s, v9.4s, %15.s[0]        \n" //out_at_0n += i20 * k01.0
                               "fmla   v14.4s, v11.4s, %15.s[1]      \n" //i21 * k01.1
                               "fmla   v15.4s, v12.4s, %15.s[2]      \n" //i22 * k01.2

                               "prfm   pldl1keep, [%6, #256]         \n"
                               "ld1    {v9.4s, v10.4s}, [%6]         \n" //input_at_3->v9.4s,v10.4s
                               "add    %6, %6, #16                   \n"

                               "fmla   v8.4s, v9.4s, %16.s[0]        \n" //output_at_0n += i30 * k02.0

                               "ext    v11.16b, v9.16b, v10.16b, #4  \n" //v9.4s[0,1,2,3],v10.4s[a,b,c,d]->v11[1,2,3,a]
                               "ext    v12.16b, v9.16b, v10.16b, #8  \n" //v9.4s[0,1,2,3],v10.4s[a,b,c,d]->v11[2,3,a,b]

                               "fmla   v14.4s, v11.4s, %16.s[1]      \n" //i31 * k02.1
                               "fmla   v15.4s, v12.4s, %16.s[2]      \n" //i32 * k02.2

                               "fadd   v7.4s, v7.4s, v6.4s           \n"

                               "prfm   pldl1keep, [%3, #256]          \n"
                               "ld1    {v9.4s, v10.4s}, [%3]         \n" //input_at_0->v9.4s,v10.4s
                               "add    %3, %3, #16                   \n"

                               "fadd   v8.4s, v8.4s, v14.4s          \n"
                               "fadd   v7.4s, v7.4s, v13.4s          \n"
                               "fadd   v8.4s, v8.4s, v15.4s          \n"

                               "ext    v11.16b, v9.16b, v10.16b, #4  \n" //v9.4s[0,1,2,3],v10.4s[a,b,c,d]->v11[1,2,3,a]
                               "ext    v12.16b, v9.16b, v10.16b, #8  \n" //v9.4s[0,1,2,3],v10.4s[a,b,c,d]->v11[2,3,a,b]

                               "st1    {v7.4s}, [%1], #16            \n"
                               "st1    {v8.4s}, [%2], #16            \n"

                               "subs   %w0, %w0, #1                  \n"
                               "bne    0b                            \n"

                               "sub    %3, %3, #16                   \n"
                               : "=r"(w_count),                        //%0
                               "=r"(out_at_0),                         //%1
                               "=r"(out_at_0n),                        //%2
                               "=r"(input_at_0),                       //%3
                               "=r"(input_at_1),                       //%4
                               "=r"(input_at_2),                       //%5
                               "=r"(input_at_3)                        //%6
                               : "0"(w_count),
                               "1"(out_at_0),
                               "2"(out_at_0n),
                               "3"(input_at_0),
                               "4"(input_at_1),
                               "5"(input_at_2),
                               "6"(input_at_3),
                               "w"(k00_x4),                      //%14
                               "w"(k01_x4),                      //%15
                               "w"(k02_x4)                       //%16
                               : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                               );
#endif //__aarch64__
#endif //_ARM_NEON__
                           }
                           input_at_0 += 2 + input_width;
                           input_at_1 += 2 + input_width;
                           input_at_2 += 2 + input_width;
                           input_at_3 += 2 + input_width;

                           out_at_0 += out_width;out_at_0n += out_width;
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