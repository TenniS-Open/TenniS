#include "kernels/opt/conv_algorithm.h"


#ifdef TS_USE_OPENMP
#include <omp.h>
#endif

namespace ts {
    namespace opt {

        template<typename T>
        static void inner_transpose_temp(size_t m, size_t n, T *in, T *out) //  A[m][n] -> A[n][m]
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    out[j * m + i] = in[i * n + j];
        }

        template<typename T>
        static void inner_cut(const Tensor& src, Tensor& dst,int cut_top, int cut_bottom, int cut_left, int cut_right) {

            auto src_shape = src.sizes();
            int num = src_shape[0];
            int channel = src_shape[1];
            int src_h = src_shape[2];
            int src_w = src_shape[3];
            int src_channel_offset = src_h * src_w;
            int src_num_offset = src_shape[1] * src_channel_offset;

            auto dst_shape = dst.sizes();
            dst_shape[0] = num;
            dst_shape[1] = channel;
            int out_h = src_h - cut_top - cut_bottom;
            int out_w = src_w - cut_left - cut_right;
            dst_shape[2] = out_h;
            dst_shape[3] = out_w;
            dst.reshape(dst_shape);
            int dst_channel_offset = out_h * out_w;
            int dst_num_offset = channel * dst_channel_offset;

            const T* src_data = src.data<T>();
            T* dst_data = dst.data<T>();

            for (int n = 0; n < num; n++){
                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (int c = 0; c < channel; c++){
                    const T* src_at = src_data + n * src_num_offset + c * src_channel_offset;
                    T* dst_at = dst_data + n * dst_num_offset + c * dst_channel_offset;

                    const T* cut_at = src_at + cut_top * src_w + cut_left;

                    for (int h = 0; h < out_h; h++){
                        if (out_w <12){
                            for (int w = 0; w < out_w; w++)
                            {
                                dst_at[w] = cut_at[w];
                            }
                        }
                        else {
                            std::memcpy(dst_at,cut_at,out_w*sizeof(T));
                        }

                        dst_at += out_w;
                        cut_at += src_w;
                    }
                }
            }
        }

        template<typename T>
        static void inner_pad(const Tensor& src, Tensor& dst, int pad_top, int pad_bottom, int pad_left, int pad_right,T pad_value) {

            auto src_shape = src.sizes();
            int num = src_shape[0];
            int channel = src_shape[1];
            int src_h = src_shape[2];
            int src_w = src_shape[3];
            int src_channel_offset = src_h * src_w;
            int src_num_offset = src_shape[1] * src_channel_offset;

            auto dst_shape = dst.sizes();
            dst_shape[0] = num;
            dst_shape[1] = channel;
            int out_h = src_h + pad_top + pad_bottom;
            int out_w = src_w + pad_left + pad_right;
            dst_shape[2] = out_h;
            dst_shape[3] = out_w;
            dst.reshape(dst_shape);
            int dst_channel_offset = out_h * out_w;
            int dst_num_offset = channel * dst_channel_offset;

            const T* src_data = src.data<T>();
            T* dst_data = dst.data<T>();

            for (int n = 0; n < num; n++)
            {
                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (int c = 0; c < channel; c++)
                {
                    const T* src_at = src_data + n * src_num_offset + c * src_channel_offset;
                    T* dst_at = dst_data + n * dst_num_offset + c * dst_channel_offset;

                    //fill top
                    int y = 0;
                    for (; y < pad_top; y++)
                    {
                        int x = 0;
                        for (; x < out_w; x++)
                        {
                            dst_at[x] = pad_value;
                        }
                        dst_at += out_w;
                    }

                    //fill center
                    for (; y < pad_top + src_shape[2]; y++)
                    {
                        int x = 0;
                        for (; x < pad_left; x++)
                        {
                            dst_at[x] = pad_value;
                        }
                        if (src_w < 12)
                        {
                            for (; x < (pad_left + src_w); x++)
                            {
                                dst_at[x] = src_at[x - pad_left];
                            }
                        }
                        else
                        {
                            std::memcpy(dst_at + pad_left, src_at, src_w * sizeof(T));
                            x += src_w;
                        }
                        for (; x < out_w; x++)
                        {
                            dst_at[x] = pad_value;
                        }
                        dst_at += out_w;
                        src_at += src_w;
                    }
                    // fill bottom
                    for ( ;y < out_h; y++)
                    {
                        int x = 0;
                        for (; x < out_w; x++)
                        {
                            dst_at[x] = pad_value;
                        }
                        dst_at += out_w;
                    }
                }
            }
        }

        template<typename T>
        void Conv2dAlgorithm<T>::conv3x3_winograd23_transform_kernel_1(const Tensor& kernel, Tensor &kernel_tm) {
            auto kernel_shape = kernel.sizes();
            int out_channel = kernel_shape[0];
            int input_channel = kernel_shape[1];

            const T* p_kernel = kernel.data<T>();
            int kernel_num_offset = input_channel * 9;
            T* p_kernel_tm = kernel_tm.data<T>();
            int kernel_tm_num_offset = input_channel * 16;

            const T ktm[4][3] = {
                { 1,     0,     0 },
                { T(1) / 2,   T(1) / 2,   T(1) / 2 },
                { T(1) / 2,   -T(1) / 2,   T(1) / 2 },
                { 0,     0,     1 }
            };

            for (int p = 0; p < out_channel; p++)
            {
                for (int q = 0; q < input_channel; q++)
                {
                    const T* kernel_at = p_kernel + p * kernel_num_offset + q * 9;
                    T* kernel_tm_at = p_kernel_tm + p * kernel_tm_num_offset + q * 16;

                    // transform kernel
                    const T* k0 = kernel_at;
                    const T* k1 = kernel_at + 3;
                    const T* k2 = kernel_at + 6;

                    T tmp[4][3];
                    for (int i = 0; i<4; i++)
                    {
                        tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                        tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                        tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
                    }

                    // UT
                    for (int j = 0; j<4; j++)
                    {
                        T* tmpp = &tmp[j][0];

                        for (int i = 0; i<4; i++)
                        {
                            kernel_tm_at[j * 4 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                        }
                    }
                }
            }
        }

        template<typename T>
        void Conv2dAlgorithm<T>::conv3x3_winograd23_transform_kernel_2(const Tensor& kernel, Tensor &kernel_tm) {

            auto kernel_shape = kernel.sizes();
            int out_channel = kernel_shape[0];
            int input_channel = kernel_shape[1];

            const T* p_kernel = kernel.data<T>();
            int kernel_num_offset = input_channel * 9;
            T* p_kernel_tm = kernel_tm.data<T>();
            int kernel_tm_num_offset = input_channel * 16;

            const T ktm[12] = {
                1,     0,     0,
                T(1) / 2,   T(1) / 2,   T(1) / 2,
                T(1) / 2,   -T(1) / 2,   T(1) / 2,
                0,     0,     1
            };

            for (int p = 0; p < out_channel; p++){
                for (int q = 0; q < input_channel; q++){
                    const T* kernel_at = p_kernel + p * kernel_num_offset + q * 9;
                    T* kernel_tm_at = p_kernel_tm + p * kernel_tm_num_offset + q * 16;

                    T tmp_mid[12], tmp_out[12];
                    #ifdef TS_USE_CBLAS
                    cblas::math<T>::gemm(blas::NoTrans, blas::NoTrans, 4, 3, 3, T(1), ktm, kernel_at,0, tmp_mid);
                    #else
                    cpu::math<T>::gemm(blas::NoTrans, blas::NoTrans, 4, 3, 3, T(1), ktm, kernel_at, 0, tmp_mid);
                    #endif
                    inner_transpose_temp<T>(4,3,tmp_mid, tmp_out);
                    //inner_transpose_temp<T>(4, 3, (T*)ktm, tmp_out);
                    #ifdef TS_USE_CBLAS
                    cblas::math<T>::gemm(blas::NoTrans, blas::NoTrans, 4, 4, 3, T(1), ktm, (const T*)tmp_out, 0, kernel_tm_at);
                    //cblas::math<T>::gemm(blas::NoTrans, blas::NoTrans, 4, 4, 3, T(1), (const T*)tmp_out,ktm, 0, kernel_tm_at);
                    #else
                    cpu::math<T>::gemm(blas::NoTrans, blas::NoTrans, 4, 3, 3, T(1), ktm, (const T*)tmp_out, 0, kernel_tm_at);
                    #endif
                }
            }
        }

        template<typename T>
        void Conv2dAlgorithm<T>::conv3x3_winograd63_transform_kernel_1(const Tensor& kernel, Tensor &kernel_tm) {
            auto kernel_shape = kernel.sizes();
            int out_channel = kernel_shape[0];
            int input_channel = kernel_shape[1];

            const T* p_kernel = kernel.data<T>();
            int kernel_num_offset = input_channel * 9;
            T* p_kernel_tm = kernel_tm.data<T>();
            int kernel_tm_num_offset = input_channel * 64;

            const T ktm[8][3] = {
                { T(1),     0,     0 },
                { -T(2) / 9,  -T(2) / 9,  -T(2) / 9 },
                { -T(2) / 9,   T(2) / 9,  -T(2) / 9 },
                { T(1) / 90,  T(1) / 45,  T(2) / 45 },
                { T(1) / 90, -T(1) / 45,  T(2) / 45 },
                { T(1) / 45,  T(1) / 90, T(1) / 180 },
                { T(1) / 45, -T(1) / 90, T(1) / 180 },
                { 0,     0,     1 }
            };

            for (int p = 0; p < out_channel; p++)
            {
                for (int q = 0; q < input_channel; q++)
                {
                    const T* kernel_at = p_kernel + p * kernel_num_offset + q * 9;
                    T* kernel_tm_at = p_kernel_tm + p * kernel_tm_num_offset + q * 64;

                    // transform kernel
                    const T* k0 = kernel_at;
                    const T* k1 = kernel_at + 3;
                    const T* k2 = kernel_at + 6;

                    T tmp[8][3];
                    for (int i = 0; i<8; i++)
                    {
                        tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                        tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                        tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
                    }

                    // U
                    for (int j = 0; j<8; j++)
                    {
                        T* tmpp = &tmp[j][0];

                        for (int i = 0; i<8; i++)
                        {
                            kernel_tm_at[j * 8 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                        }
                    }
                }
            }
        }

        template<typename T>
        void Conv2dAlgorithm<T>::conv3x3_winograd23(const Tensor &x, const Tensor &k_tm, Tensor &out) {

            auto input_shape = x.sizes();
            auto k_tm_shape = k_tm.sizes();
            auto out_shape = out.sizes();

            int input_h = input_shape[2];
            int input_w = input_shape[3];
            int input_channel = input_shape[1];
            int num = input_shape[0];

            int output_h = out_shape[2];
            int output_w = out_shape[3];
            int output_channel = out_shape[1];

            //pad
            output_w = (output_w + 1) / 2 * 2;
            output_h = (output_h + 1) / 2 * 2;

            int input_padded_w = output_w + 2;  //output_w = (input_w - 3)/1 - 1;
            int input_padded_h = output_h + 2;  //output_h = (input_h - 3)/1 - 1;

            Shape input_bordered_s = { num, input_channel, input_padded_h, input_padded_w };
            Tensor input_bordered(MemoryDevice(CPU), x.dtype(), input_bordered_s);
            int bordered_c_offset = input_padded_h * input_padded_w;
            int bordered_num_offset = input_channel * bordered_c_offset;

            inner_pad<T>(x, input_bordered, 0, input_padded_h - input_h, 0, input_padded_w - input_w, 0);

            //transform input data

            // const float BT[4][4] = {
            //     {1.0f,  0.0f, -1.0f,  0.0f},
            //     {0.0f,  1.0f,  1.00f, 0.0f},
            //     {0.0f, -1.0f,  1.00f, 0.0f},
            //     {0.0f, -1.0f,  0.00f, 1.0f}
            // };   

            int w_tm = output_w / 2 * 4;
            int h_tm = output_h / 2 * 4;
            int col_blocks = w_tm / 4;
            int row_blocks = h_tm / 4;
            int num_blocks = col_blocks * row_blocks;
            Shape input_tm_s = { num, input_channel, num_blocks, 16 };
            Tensor input_tm(MemoryDevice(CPU), x.dtype(), input_tm_s);
            int tm_c_offset = 16 * num_blocks;
            int tm_num_offset = input_channel * tm_c_offset;

            const T* src_ptr = input_bordered.data<T>();
            T* dst_ptr = input_tm.data<T>();
            for (int n = 0; n < num; n++)
            {
                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (int c = 0; c < input_channel; c++)
                {
                    const T* src_at = src_ptr + n * bordered_num_offset + c * bordered_c_offset;
                    T* dst_at = dst_ptr + n * tm_num_offset + c * tm_c_offset;

                    for (int i = 0; i < col_blocks; i++)
                    {
                        const T* r0 = src_at + i * input_padded_w * 2;
                        const T* r1 = r0 + input_padded_w;
                        const T* r2 = r1 + input_padded_w;
                        const T* r3 = r2 + input_padded_w;

                        for (int j = 0; j < row_blocks; j++)
                        {
                            T d0[4], d1[4], d2[4], d3[4];
                            T w0[4], w1[4], w2[4], w3[4];
                            T t0[4], t1[4], t2[4], t3[4];

                            for (int n = 0; n < 4; n++)
                            {
                                d0[n] = r0[n];
                                d1[n] = r1[n];
                                d2[n] = r2[n];
                                d3[n] = r3[n];
                            }
                            
                            // BT * d * B == (BT * (BT*d)T)T

                            // w = BT * d
                            for (int n = 0; n < 4; n++)
                            {
                                w0[n] = d0[n] - d2[n];
                                w1[n] = d1[n] + d2[n];
                                w2[n] = d2[n] - d1[n];
                                w3[n] = d3[n] - d1[n];
                            }

                            // w to wT
                            {
                                t0[0] = w0[0]; t1[0] = w0[1]; t2[0] = w0[2]; t3[0] = w0[3];
                                t0[1] = w1[0]; t1[1] = w1[1]; t2[1] = w1[2]; t3[1] = w1[3];
                                t0[2] = w2[0]; t1[2] = w2[1]; t2[2] = w2[2]; t3[2] = w2[3];
                                t0[3] = w3[0]; t1[3] = w3[1]; t2[3] = w3[2]; t3[3] = w3[3];
                            }

                            // d = BT * wT 
                            for (int n = 0; n < 4; n++)
                            {
                                d0[n] = t0[n] - t2[n];
                                d1[n] = t1[n] + t2[n];
                                d2[n] = t2[n] - t1[n];
                                d3[n] = t3[n] - t1[n];
                            }

                            for (int n = 0; n < 4; n++)
                            {
                                dst_at[n] = d0[n];
                                dst_at[n + 4] = d1[n];
                                dst_at[n + 8] = d2[n];
                                dst_at[n + 12] = d3[n];
                            }

                            r0 += 2;
                            r1 += 2;
                            r2 += 2;
                            r3 += 2;

                            dst_at += 16;

                        }
                    }
                }

            }

            //begin dot
            Shape out_tm_s = { num, output_channel, num_blocks, 16 };
            Tensor output_tm(MemoryDevice(CPU), x.dtype(), out_tm_s);
            int outtm_c_offset = num_blocks * 16;
            int outtm_n_offset = output_channel * outtm_c_offset;

            int ktm_c_offset = k_tm_shape[2] * k_tm_shape[3];
            int ktm_n_offset = k_tm_shape[1] * ktm_c_offset;

            int outch = output_channel >> 2;
            int remain_outch = outch << 2;

            T* out_tm_ptr = output_tm.data<T>();
            T* input_tm_ptr = dst_ptr;

            for (int n = 0; n < num; n++)
            {
                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (int cc = 0; cc < outch; cc++)
                {
                    int c = cc * 4;

                    T* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                    T* out_tm_1 = out_tm_0 + outtm_c_offset;
                    T* out_tm_2 = out_tm_1 + outtm_c_offset;
                    T* out_tm_3 = out_tm_2 + outtm_c_offset;

                    const T* kernel_tm_ptr = k_tm.data<T>();

                    const T* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;
                    const T* kernel_tm_1 = kernel_tm_0 + ktm_n_offset;
                    const T* kernel_tm_2 = kernel_tm_1 + ktm_n_offset;
                    const T* kernel_tm_3 = kernel_tm_2 + ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        T* out_0 = out_tm_0 + i * 16;
                        T* out_1 = out_tm_1 + i * 16;
                        T* out_2 = out_tm_2 + i * 16;
                        T* out_3 = out_tm_3 + i * 16;

                        T sum_0[16] = { T(0) };
                        T sum_1[16] = { T(0) };
                        T sum_2[16] = { T(0) };
                        T sum_3[16] = { T(0) };

                        int inputch = input_channel >> 2;
                        int remain_inputch = inputch << 2;
                        for (int qq = 0; qq < inputch; qq++)
                        {
                            int q = qq * 4;
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 16;
                            const T* r1 = r0 + tm_c_offset;
                            const T* r2 = r1 + tm_c_offset;
                            const T* r3 = r2 + tm_c_offset;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = kernel_tm_1 + q * ktm_c_offset;
                            const T* k2 = kernel_tm_2 + q * ktm_c_offset;
                            const T* k3 = kernel_tm_3 + q * ktm_c_offset;

                            for (int k = 0; k < 16; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                k0 += 16;
                                sum_0[k] += r1[k] * k0[k];
                                k0 += 16;
                                sum_0[k] += r2[k] * k0[k];
                                k0 += 16;
                                sum_0[k] += r3[k] * k0[k];
                                k0 -= 48;

                                sum_1[k] += r0[k] * k1[k];
                                k1 += 16;
                                sum_1[k] += r1[k] * k1[k];
                                k1 += 16;
                                sum_1[k] += r2[k] * k1[k];
                                k1 += 16;
                                sum_1[k] += r3[k] * k1[k];
                                k1 -= 48;

                                sum_2[k] += r0[k] * k2[k];
                                k2 += 16;
                                sum_2[k] += r1[k] * k2[k];
                                k2 += 16;
                                sum_2[k] += r2[k] * k2[k];
                                k2 += 16;
                                sum_2[k] += r3[k] * k2[k];
                                k2 -= 48;

                                sum_3[k] += r0[k] * k3[k];
                                k3 += 16;
                                sum_3[k] += r1[k] * k3[k];
                                k3 += 16;
                                sum_3[k] += r2[k] * k3[k];
                                k3 += 16;
                                sum_3[k] += r3[k] * k3[k];
                                k3 -= 48;
                            }
                        }

                        for (int q = remain_inputch; q < input_channel; q++)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 16;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = kernel_tm_1 + q * ktm_c_offset;
                            const T* k2 = kernel_tm_2 + q * ktm_c_offset;
                            const T* k3 = kernel_tm_3 + q * ktm_c_offset;

                            for (int k = 0; k < 16; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                sum_1[k] += r0[k] * k1[k];
                                sum_2[k] += r0[k] * k2[k];
                                sum_3[k] += r0[k] * k3[k];
                            }
                        }

                        for (int k = 0; k < 16; k++)
                        {
                            out_0[k] = sum_0[k];
                            out_1[k] = sum_1[k];
                            out_2[k] = sum_2[k];
                            out_3[k] = sum_3[k];
                        }

                    }
                }

                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (int c = remain_outch; c < output_channel; c++)
                {
                    T* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;

                    const T* kernel_tm_ptr = k_tm.data<T>();
                    const T* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        T* out_0 = out_tm_0 + i * 16;
                        T sum_0[16] = { T(0) };

                        int q = 0;
                        for (; q + 3 < input_channel; q += 4)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 16;
                            const T* r1 = r0 + tm_c_offset;
                            const T* r2 = r1 + tm_c_offset;
                            const T* r3 = r2 + tm_c_offset;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = k0 + ktm_c_offset;
                            const T* k2 = k1 + ktm_c_offset;
                            const T* k3 = k2 + ktm_c_offset;

                            for (int k = 0; k < 16; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                sum_0[k] += r1[k] * k1[k];
                                sum_0[k] += r2[k] * k2[k];
                                sum_0[k] += r3[k] * k3[k];
                            }
                        }

                        for (; q < input_channel; q++)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 16;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;

                            for (int k = 0; k < 16; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                            }
                        }

                        for (int k = 0; k < 16; k++)
                        {
                            out_0[k] = sum_0[k];
                        }

                    }
                }
            }

            //begin transform output
            Shape output_bordered_s = { num, output_channel, output_h, output_w };
            Tensor output_bordered(MemoryDevice(CPU), out.dtype(), output_bordered_s);
            int outbo_c_offset = output_h * output_w;
            int outbo_n_offset = output_channel * outbo_c_offset;

            T* out_ptr = output_bordered.data<T>();

            // const float AT[2][4] = {
            //     {1.0f,  1.0f,  1.0f,  0.0f},
            //     {0.0f,  1.0f, -1.0f,  1.0f}
            // }; 

            for (int n = 0; n < num; n++)
            {
                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (int c = 0; c < output_channel; c++)
                {
                    T* output_tm_at = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                    T* out_at = out_ptr + n * outbo_n_offset + c * outbo_c_offset;

                    for (int i = 0; i < col_blocks; i++)
                    {
                        T* out_0 = out_at + i * output_w * 2;
                        T* out_1 = out_0 + output_w;

                        for (int j = 0; j < row_blocks; j++)
                        {
                            T* out_tile = output_tm_at + (i * col_blocks + j) * 16;

                            float s0[4], s1[4], s2[4], s3[4];
                            float w0[4], w1[4];
                            float d0[2], d1[2], d2[2], d3[2];
                            float o0[2], o1[2];

                            for (int n = 0; n < 4; n++)
                            {
                                s0[n] = out_tile[n];
                                s1[n] = out_tile[n + 4];
                                s2[n] = out_tile[n + 8];
                                s3[n] = out_tile[n + 12];
                            }
                            // w = A_T * W
                            for (int n = 0; n < 4; n++)
                            {
                                w0[n] = s0[n] + s1[n] + s2[n];
                                w1[n] = s1[n] - s2[n] + s3[n];
                            }
                            // transpose w to w_t
                            {
                                d0[0] = w0[0]; d0[1] = w1[0];
                                d1[0] = w0[1]; d1[1] = w1[1];
                                d2[0] = w0[2]; d2[1] = w1[2];
                                d3[0] = w0[3]; d3[1] = w1[3];
                            }
                            // Y = A_T * w_t
                            for (int n = 0; n < 2; n++)
                            {
                                o0[n] = d0[n] + d1[n] + d2[n];
                                o1[n] = d1[n] - d2[n] + d3[n];
                            }

                            out_0[0] = o0[0];
                            out_0[1] = o0[1];
                            out_1[0] = o1[0];
                            out_1[1] = o1[1];

                            out_0 += 2;
                            out_1 += 2;
                        }
                    }
                }
            }

            inner_cut<T>(output_bordered, out, 0, output_h - out_shape[2], 0, output_w - out_shape[3]);

        }

        template<typename T>
        void Conv2dAlgorithm<T>::conv3x3_winograd63_transform_kernel_2(const Tensor& kernel, Tensor &kernel_tm) {

            auto kernel_shape = kernel.sizes();
            int out_channel = kernel_shape[0];
            int input_channel = kernel_shape[1];

            const T* p_kernel = kernel.data<T>();
            int kernel_num_offset = input_channel * 9;
            T* p_kernel_tm = kernel_tm.data<T>();
            int kernel_tm_num_offset = input_channel * 64;

            const T ktm[24] =
            {
                1, 0, 0,
                -T(2) / 9, -T(2) / 9, -T(2) / 9,
                -T(2) / 9, T(2) / 9, -T(2) / 9,
                T(1) / 90, T(1) / 45, T(2) / 45,
                T(1) / 90, -T(1) / 45, T(2) / 45,
                T(1) / 45, T(1) / 90, T(1) / 180,
                T(1) / 45, -T(1) / 90, T(1) / 180,
                0, 0, 1
            };

            for (int p = 0; p < out_channel; p++) {
                for (int q = 0; q < input_channel; q++) {
                    const T* kernel_at = p_kernel + p * kernel_num_offset + q * 9;
                    T* kernel_tm_at = p_kernel_tm + p * kernel_tm_num_offset + q * 64;

                    T tmp_mid[24], tmp_out[24];
                    #ifdef TS_USE_CBLAS
                    cblas::math<T>::gemm(blas::NoTrans, blas::NoTrans, 8, 3, 3, T(1), ktm, kernel_at, 0, tmp_mid);
                    #else
                    cpu::math<T>::gemm(blas::NoTrans, blas::NoTrans, 8, 3, 3, T(1), ktm, kernel_at, 0, tmp_mid);
                    #endif
                    inner_transpose_temp<T>(8, 3, tmp_mid, tmp_out);
                    //inner_transpose_temp<T>(4, 3, (T*)ktm, tmp_out);
                    #ifdef TS_USE_CBLAS
                    cblas::math<T>::gemm(blas::NoTrans, blas::NoTrans, 8, 8, 3, T(1), ktm, (const T*)tmp_out, 0, kernel_tm_at);
                    //cblas::math<T>::gemm(blas::NoTrans, blas::NoTrans, 4, 4, 3, T(1), (const T*)tmp_out,ktm, 0, kernel_tm_at);
                    #else
                    cpu::math<T>::gemm(blas::NoTrans, blas::NoTrans, 8, 8, 3, T(1), ktm, (const T*)tmp_out, 0, kernel_tm_at);
                    #endif
                }
            }
        }

        template<typename T>
        void Conv2dAlgorithm<T>::conv3x3_winograd63(const Tensor &x, const Tensor &k_tm, Tensor &out) {
            
            auto input_shape = x.sizes();
            auto k_tm_shape = k_tm.sizes();
            auto out_shape = out.sizes();

            int input_h = input_shape[2];
            int input_w = input_shape[3];
            int input_channel = input_shape[1];
            int num = input_shape[0];

            int output_h = out_shape[2];
            int output_w = out_shape[3];
            int output_channel = out_shape[1];

            //pad
            output_w = (output_w + 5) / 6 * 6;
            output_h = (output_h + 5) / 6 * 6;

            int input_padded_w = output_w + 2;  //output_w = (input_w - 3)/1 - 1;
            int input_padded_h = output_h + 2;  //output_h = (input_h - 3)/1 - 1;

            Shape input_bordered_s = { num, input_channel, input_padded_h, input_padded_w };
            Tensor input_bordered(MemoryDevice(CPU), x.dtype(), input_bordered_s);
            int bordered_c_offset = input_padded_h * input_padded_w;
            int bordered_num_offset = input_channel * bordered_c_offset;

            inner_pad<T>(x, input_bordered, 0, input_padded_h-input_h, 0, input_padded_w-input_w,0);

            //transform input data

            //const float BT[8][8] = {
            //    {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
            //
            //    {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
            //    {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
            //
            //    {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
            //    {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
            //
            //    {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
            //    {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
            //
            //    {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
            //};

            int w_tm = output_w / 6 * 8;
            int h_tm = output_h / 6 * 8;
            int col_blocks = w_tm / 8;
            int row_blocks = h_tm / 8;
            int num_blocks = col_blocks * row_blocks;
            Shape input_tm_s = {num, input_channel, num_blocks, 64 };
            Tensor input_tm(MemoryDevice(CPU), x.dtype(), input_tm_s);
            int tm_c_offset = 64 * num_blocks;
            int tm_num_offset = input_channel * tm_c_offset;

            const T* src_ptr = input_bordered.data<T>();
            T* dst_ptr = input_tm.data<T>();
            for (int n = 0; n < num; n++)
            {
                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (int c = 0; c < input_channel; c++)
                {
                    const T* src_at = src_ptr + n * bordered_num_offset + c * bordered_c_offset;
                    T* dst_at = dst_ptr + n * tm_num_offset + c * tm_c_offset;

                    T tmp[8][8];//save (d*B)T
                    for (int i = 0; i < col_blocks; i++)
                    {
                        for (int j = 0; j < row_blocks; j++)
                        {
                            const T* r0 = src_at + i * input_padded_w * 6 + j * 6;

                            for (int m = 0; m < 8; m++)
                            {
                                tmp[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25f;
                                tmp[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25f;

                                float tmp12_a = (r0[2] + r0[6] - r0[4] * 4.25f);
                                float tmp12_b = (r0[1] + r0[5] - r0[3] * 4.25f);

                                tmp[1][m] = tmp12_a + tmp12_b;
                                tmp[2][m] = tmp12_a - tmp12_b;

                                float tmp34_a = (r0[6] + r0[2] * 0.25f - r0[4] * 1.25f);
                                float tmp34_b = (r0[1] * 0.5f - r0[3] * 2.5f + r0[5] * 2.f);

                                tmp[3][m] = tmp34_a + tmp34_b;
                                tmp[4][m] = tmp34_a - tmp34_b;

                                float tmp56_a = (r0[6] + (r0[2] - r0[4] * 1.25f) * 4.f);
                                float tmp56_b = (r0[1] * 2.f - r0[3] * 2.5f + r0[5] * 0.5f);

                                tmp[5][m] = tmp56_a + tmp56_b;
                                tmp[6][m] = tmp56_a - tmp56_b;

                                r0 += input_padded_w;
                            }

                            T* d0 = dst_at + (i * col_blocks + j) * 64;

                            T* d1 = d0 + 1;
                            T* d2 = d1 + 1;
                            T* d3 = d2 + 1;
                            T* d4 = d3 + 1;
                            T* d5 = d4 + 1;
                            T* d6 = d5 + 1;
                            T* d7 = d6 + 1;

                            //(d*B)T * B == (BT*d*B)T == VT
                            for (int m = 0; m < 8; m++)
                            {
                                const T* tmp0 = tmp[m];

                                d0[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25f;
                                d7[0] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25f;

                                float tmp12_a = (tmp0[2] + tmp0[6] - tmp0[4] * 4.25f);
                                float tmp12_b = (tmp0[1] - tmp0[3] * 4.25f + tmp0[5]);

                                d1[0] = tmp12_a + tmp12_b;
                                d2[0] = tmp12_a - tmp12_b;

                                float tmp34_a = (tmp0[6] + tmp0[2] * 0.25f - tmp0[4] * 1.25f);
                                float tmp34_b = (tmp0[1] * 0.5f - tmp0[3] * 2.5f + tmp0[5] * 2.f);

                                d3[0] = tmp34_a + tmp34_b;
                                d4[0] = tmp34_a - tmp34_b;

                                float tmp56_a = (tmp0[6] + (tmp0[2] - tmp0[4] * 1.25f) * 4.f);
                                float tmp56_b = (tmp0[1] * 2.f - tmp0[3] * 2.5f + tmp0[5] * 0.5f);

                                d5[0] = tmp56_a + tmp56_b;
                                d6[0] = tmp56_a - tmp56_b;

                                d0 += 8;
                                d1 += 8;
                                d2 += 8;
                                d3 += 8;
                                d4 += 8;
                                d5 += 8;
                                d6 += 8;
                                d7 += 8;

                            }
                        }
                    }
                }

            }

            //begin dot
            Shape out_tm_s = { num, output_channel, num_blocks, 64 };
            Tensor output_tm(MemoryDevice(CPU), x.dtype(), out_tm_s);
            int outtm_c_offset = num_blocks * 64;
            int outtm_n_offset = output_channel * outtm_c_offset;

            int ktm_c_offset = k_tm_shape[2] * k_tm_shape[3];
            int ktm_n_offset = k_tm_shape[1] * ktm_c_offset;

            int outch = output_channel >> 2;
            int remain_outch = outch << 2;

            T* out_tm_ptr = output_tm.data<T>();
            T* input_tm_ptr = dst_ptr;

            for (int n = 0; n < num; n++)
            {
                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (int cc = 0; cc < outch; cc++)
                {
                    int c = cc * 4;

                    T* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                    T* out_tm_1 = out_tm_0 + outtm_c_offset;
                    T* out_tm_2 = out_tm_1 + outtm_c_offset;
                    T* out_tm_3 = out_tm_2 + outtm_c_offset;

                    const T* kernel_tm_ptr = k_tm.data<T>();

                    const T* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;
                    const T* kernel_tm_1 = kernel_tm_0 + ktm_n_offset;
                    const T* kernel_tm_2 = kernel_tm_1 + ktm_n_offset;
                    const T* kernel_tm_3 = kernel_tm_2 + ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        T* out_0 = out_tm_0 + i * 64;
                        T* out_1 = out_tm_1 + i * 64;
                        T* out_2 = out_tm_2 + i * 64;
                        T* out_3 = out_tm_3 + i * 64;

                        T sum_0[64] = { T(0) };
                        T sum_1[64] = { T(0) };
                        T sum_2[64] = { T(0) };
                        T sum_3[64] = { T(0) };

                        int inputch = input_channel >> 2;
                        int remain_inputch = inputch << 2;
                        for (int qq = 0; qq < inputch; qq++)
                        {
                            int q = qq * 4;
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 64;
                            const T* r1 = r0 + tm_c_offset;
                            const T* r2 = r1 + tm_c_offset;
                            const T* r3 = r2 + tm_c_offset;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = kernel_tm_1 + q * ktm_c_offset;
                            const T* k2 = kernel_tm_2 + q * ktm_c_offset;
                            const T* k3 = kernel_tm_3 + q * ktm_c_offset;

                            for (int k = 0; k < 64; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                k0 += 64;
                                sum_0[k] += r1[k] * k0[k];
                                k0 += 64;
                                sum_0[k] += r2[k] * k0[k];
                                k0 += 64;
                                sum_0[k] += r3[k] * k0[k];
                                k0 -= 192;

                                sum_1[k] += r0[k] * k1[k];
                                k1 += 64;
                                sum_1[k] += r1[k] * k1[k];
                                k1 += 64;
                                sum_1[k] += r2[k] * k1[k];
                                k1 += 64;
                                sum_1[k] += r3[k] * k1[k];
                                k1 -= 192;

                                sum_2[k] += r0[k] * k2[k];
                                k2 += 64;
                                sum_2[k] += r1[k] * k2[k];
                                k2 += 64;
                                sum_2[k] += r2[k] * k2[k];
                                k2 += 64;
                                sum_2[k] += r3[k] * k2[k];
                                k2 -= 192;

                                sum_3[k] += r0[k] * k3[k];
                                k3 += 64;
                                sum_3[k] += r1[k] * k3[k];
                                k3 += 64;
                                sum_3[k] += r2[k] * k3[k];
                                k3 += 64;
                                sum_3[k] += r3[k] * k3[k];
                                k3 -= 192;
                            }
                        }
                        for (int q = remain_inputch; q < input_channel; q++)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 64;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = kernel_tm_1 + q * ktm_c_offset;
                            const T* k2 = kernel_tm_2 + q * ktm_c_offset;
                            const T* k3 = kernel_tm_3 + q * ktm_c_offset;

                            for (int k = 0; k < 64; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                sum_1[k] += r0[k] * k1[k];
                                sum_2[k] += r0[k] * k2[k];
                                sum_3[k] += r0[k] * k3[k];
                            }
                        }

                        for (int k = 0; k < 64; k++)
                        {
                            out_0[k] = sum_0[k];
                            out_1[k] = sum_1[k];
                            out_2[k] = sum_2[k];
                            out_3[k] = sum_3[k];
                        }

                    }
                }

                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (int c = remain_outch; c < output_channel; c++)
                {
                    T* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;

                    const T* kernel_tm_ptr = k_tm.data<T>();
                    const T* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        T* out_0 = out_tm_0 + i * 64;
                        T sum_0[64] = { T(0) };

                        int q = 0;
                        for (; q+3 < input_channel; q+=4)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 64;
                            const T* r1 = r0 + tm_c_offset;
                            const T* r2 = r1 + tm_c_offset;
                            const T* r3 = r2 + tm_c_offset;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = k0 + ktm_c_offset;
                            const T* k2 = k1 + ktm_c_offset;
                            const T* k3 = k2 + ktm_c_offset;

                            for (int k = 0; k < 64; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                sum_0[k] += r0[k] * k1[k];
                                sum_0[k] += r0[k] * k2[k];
                                sum_0[k] += r0[k] * k3[k];
                            }
                        }

                        for (; q < input_channel; q++)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 64;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;

                            for (int k = 0; k < 64; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                            }
                        }

                        for (int k = 0; k < 64; k++)
                        {
                            out_0[k] = sum_0[k];
                        }

                    }
                }
            }

            //begin transform output
            Shape output_bordered_s = { num, output_channel, output_h, output_w };
            Tensor output_bordered(MemoryDevice(CPU), out.dtype(), output_bordered_s);
            int outbo_c_offset = output_h * output_w;
            int outbo_n_offset = output_channel * outbo_c_offset;

            T* out_ptr = output_bordered.data<T>();

            //const float AT[6][8] = {
            //    {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
            //    {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
            //    {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
            //    {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
            //    {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
            //    {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
            //};

            // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
            // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
            // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
            // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
            // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
            // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

            //reuse (r1 + r2) (r1 - r2) (r3 + r4) (r3 - r4) (r5 + r6) (r5 - r6)

            for (int n = 0; n < num; n++)
            {
                #pragma omp parallel for num_threads(omp_get_max_threads())
                for (int c = 0; c < output_channel; c++)
                {
                    T* output_tm_at = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                    T* out_at = out_ptr + n * outbo_n_offset + c * outbo_c_offset;

                    T tmp[6][8]; //(WT*A)T == AT*W
                    for (int i = 0; i < col_blocks; i++)
                    {
                        for (int j = 0; j < row_blocks; j++)
                        {
                            const T* w0 = output_tm_at + (i * col_blocks + j) * 64;

                            for (int m = 0; m < 8; m++)
                            {
                                T tmp1add2 = w0[1] + w0[2];
                                T tmp1sub2 = w0[1] - w0[2];
                                T tmp3add4 = w0[3] + w0[4];
                                T tmp3sub4 = w0[3] - w0[4];
                                T tmp5add6 = w0[5] + w0[6];
                                T tmp5sub6 = w0[5] - w0[6];

                                tmp[0][m] = w0[0] + tmp1add2 + tmp3add4 + tmp5add6 * 32;
                                tmp[1][m] = tmp1sub2 + tmp3sub4 * 2 + tmp5sub6 * 16;
                                tmp[2][m] = tmp1add2 + tmp3add4 * 4 + tmp5add6 * 8;
                                tmp[3][m] = tmp1sub2 + tmp3sub4 * 8 + tmp5sub6 * 4;
                                tmp[4][m] = tmp1add2 + tmp3add4 * 16 + tmp5add6 * 2;
                                tmp[5][m] = tmp1sub2 + tmp3sub4 * 32 + tmp5sub6 + w0[7];

                                //tmp[0][m] = w0[0] + (w0[1] + w0[2]) + (w0[3] + w0[4]) + (w0[5] + w0[6]) * 32;
                                //tmp[1][m] = (w0[1] - w0[2]) + (w0[3] - w0[4]) * 2 + (w0[5] - w0[6]) * 16;
                                //tmp[2][m] = (w0[1] + w0[2]) + (w0[3] + w0[4]) * 4 + (w0[5] + w0[6]) * 8;
                                //tmp[3][m] = (w0[1] - w0[2]) + (w0[3] - w0[4]) * 8 + (w0[5] - w0[6]) * 4;
                                //tmp[4][m] = (w0[1] + w0[2]) + (w0[3] + w0[4]) * 16 + (w0[5] + w0[6]) * 2;
                                //tmp[5][m] = (w0[1] - w0[2]) + (w0[3] - w0[4]) * 32 + (w0[5] - w0[6]) + w0[7];

                                w0 += 8;
                            }

                            T* d0 = out_at + i * output_w * 6 + j * 6;
                            T* d1 = d0 + 1;
                            T* d2 = d1 + 1;
                            T* d3 = d2 + 1;
                            T* d4 = d3 + 1;
                            T* d5 = d4 + 1;

                            for (int m = 0; m < 6; m++)
                            {
                                const T* tmp0 = tmp[m];

                                d0[0] = tmp0[0] + (tmp0[1] + tmp0[2]) + (tmp0[3] + tmp0[4]) + (tmp0[5] + tmp0[6]) * 32;
                                d1[0] = (tmp0[1] - tmp0[2]) + (tmp0[3] - tmp0[4]) * 2 + (tmp0[5] - tmp0[6]) * 16;
                                d2[0] = (tmp0[1] + tmp0[2]) + (tmp0[3] + tmp0[4]) * 4 + (tmp0[5] + tmp0[6]) * 8;
                                d3[0] = (tmp0[1] - tmp0[2]) + (tmp0[3] - tmp0[4]) * 8 + (tmp0[5] - tmp0[6]) * 4;
                                d4[0] = (tmp0[1] + tmp0[2]) + (tmp0[3] + tmp0[4]) * 16 + (tmp0[5] + tmp0[6]) * 2;
                                d5[0] = (tmp0[1] - tmp0[2]) + (tmp0[3] - tmp0[4]) * 32 + (tmp0[5] - tmp0[6]) + tmp0[7]; 

                                d0 += output_w;
                                d1 += output_w;
                                d2 += output_w;
                                d3 += output_w;
                                d4 += output_w;
                                d5 += output_w;

                            }

                        }
                    }
                }
            }

            inner_cut<T>(output_bordered, out, 0, output_h - out_shape[2], 0, output_w - out_shape[3]);

        }
    }

}

template class ts::opt::Conv2dAlgorithm<ts::dtype<ts::FLOAT32>::declare>;
template class ts::opt::Conv2dAlgorithm<ts::dtype<ts::FLOAT64>::declare>;