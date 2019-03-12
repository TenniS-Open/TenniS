#include <kernels/gpu/conv2d_core.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>



namespace ts {
    namespace gpu {

        template <typename T>
        static __global__ void gpu_im2col_kernel(const int n, const T* data_im,
            const int height, const int width, const int kernel_h, const int kernel_w,
            const int pad_top, const int pad_left,
            const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w,
            const int height_col, const int width_col,
            T* data_col, T padding_value) {
            for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
                const int h_index = index / width_col;
                const int h_col = h_index % height_col;
                const int w_col = index % width_col;
                const int c_im = h_index / height_col;
                const int c_col = c_im * kernel_h * kernel_w;
                const int h_offset = h_col * stride_h - pad_top;
                const int w_offset = w_col * stride_w - pad_left;
                T* data_col_ptr = data_col;
                data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
                const T* data_im_ptr = data_im;
                data_im_ptr += (c_im * height + h_offset) * width + w_offset;
                for (int i = 0; i < kernel_h; ++i) {
                    for (int j = 0; j < kernel_w; ++j) {
                        int h_im = h_offset + i * dilation_h;
                        int w_im = w_offset + j * dilation_w;
                        *data_col_ptr =
                            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                            data_im_ptr[i * dilation_h * width + j * dilation_w] : padding_value;
                        data_col_ptr += height_col * width_col;
                    }
                }
            }
        }

        template<typename T>
        static __global__ void gpu_conv2d_compute_run_kernel(int m, int n, int k, const T *A, const T *B, T *C) {
            __shared__ T ds_A[TRANS_BLOCK_DIM][TRANS_BLOCK_DIM];
            __shared__ T ds_B[TRANS_BLOCK_DIM][TRANS_BLOCK_DIM];

            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;

            int Row = by * TRANS_BLOCK_DIM + ty;
            int Col = bx * TRANS_BLOCK_DIM + tx;

            T comp = 0;
            T Cvalue = 0;

            for (int t=0; t<(n-1)/TRANS_BLOCK_DIM+1; ++t) {
                if (Row < m && t * TRANS_BLOCK_DIM + tx < n)
                    ds_A[tx][ty] = A[Row*n+t*TRANS_BLOCK_DIM+tx];
                else
                    ds_A[tx][ty] = 0.0;

                if (t * TRANS_BLOCK_DIM + ty < n && Col < k)
                    ds_B[tx][ty] = B[(t*TRANS_BLOCK_DIM + ty)*k+Col];
                else
                    ds_B[tx][ty] = 0.0;

                __syncthreads();

                for (int i = 0; i < TRANS_BLOCK_DIM; ++i) {
                    //Cvalue += ds_A[i][ty] * ds_B[tx][i];
                    T t;
                    comp -= ds_A[i][ty] * ds_B[tx][i];
                    t = Cvalue - comp;
                    comp = (t - Cvalue) + comp;
                    Cvalue = t;
                }

                __syncthreads();

                if(Row < m && Col < k) {
                    C[Row*k+Col]=Cvalue;
                }
            }//end for
        }


        template<typename T>
        static void gpu_conv2d_nchw_compute_run(const Tensor &x, const Padding2D &padding, float padding_value,
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

            T *col_buffer = nullptr;
        
            Tensor col_tensor;
            bool is_1x1_conv = stride.height == 1 && stride.width == 1 &&
                               ksize.height == 1 && ksize.width == 1 &&
                               padding.top == 0 && padding.bottom == 0 &&
                               padding.left == 0 && padding.right == 0;

            int put_param = input_channels * output_shape[2]  * output_shape[3];
            // 1x1 conv do not need im2col
            if (!is_1x1_conv) {

                Shape tmpshape;
                tmpshape.resize(1);
                tmpshape[0] = col_buffer_size;
                col_tensor = Tensor(out.device(), out.dtype(), tmpshape);
                col_buffer = col_tensor.data<T>();
 
            }
                  
            dim3 blocksize(CUDA_BLOCK(conv_out_spatial_dim, TRANS_BLOCK_DIM), CUDA_BLOCK(weight_shape[0], TRANS_BLOCK_DIM),1);
            dim3 threadsize(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM,1);

            for(int i=0; i<number; i++) { 
                if(!is_1x1_conv) {
                    gpu_im2col_kernel<T> <<< CUDA_BLOCK(put_param, CUDA_THREAD_NUM), CUDA_THREAD_NUM >>> (
                                            put_param, pinput, input.height, input.width, 
                                            ksize.height, ksize.width, padding.top, padding.left,
                                            stride.height, stride.width, dilation.height, dilation.width,
                                            output_shape[2], output_shape[3], col_buffer, T(padding_value));

                }else {
                    col_buffer = const_cast<T *>(pinput);
                }

                gpu_conv2d_compute_run_kernel<T> <<<blocksize, threadsize>>>
                      (weight_shape[0], kernel_dims,conv_out_spatial_dim, pweight, col_buffer, poutput);
                pinput += input_number_offset;
                poutput += output_number_offset;
            }//end for

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
        case DTYPE: { gpu_conv2d_nchw_compute_run<TYPE>(x, padding, padding_value, w, stride, dilation, out, stack);; break; }
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
