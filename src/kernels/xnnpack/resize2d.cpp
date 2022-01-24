#include "resize2d.h"
#include "runtime/runtime.h"
#include <core/tensor_builder.h>
#include <memory>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>


namespace ts {
    namespace xnn {
        void Resize2D::init() {
            supper::init();

            auto ctx = ctx::get<RuntimeContext>();
            m_pool = ctx->get_xnn_threadpool();
        }

        template<typename T>
        struct ResizeImageLinear_kernel_context {
            int dst_width;
            double lfx_scl;
            double lfy_scl;
            double bias_x;
            double bias_y;
            int src_width;
            int src_height;
            int channels;
            T *dst_im;
            const T *src_im;
        };

        template<typename T>
        static inline void ResizeImageLinear_kernel(struct ResizeImageLinear_kernel_context<T> *context, size_t n_y_d) {
            for (int n_x_d = 0; n_x_d < context->dst_width; n_x_d++) {
                double lf_x_s = context->lfx_scl * n_x_d + context->bias_x;
                double lf_y_s = context->lfy_scl * n_y_d + context->bias_y;

                lf_x_s = lf_x_s >= 0 ? lf_x_s : 0;
                lf_x_s = lf_x_s < context->src_width - 1 ? lf_x_s : context->src_width - 1 - 1e-5;
                lf_y_s = lf_y_s >= 0 ? lf_y_s : 0;
                lf_y_s = lf_y_s < context->src_height - 1 ? lf_y_s : context->src_height - 1 - 1e-5;

                int n_x_s = int(lf_x_s);
                int n_y_s = int(lf_y_s);

                double lf_weight_x = lf_x_s - n_x_s;
                double lf_weight_y = lf_y_s - n_y_s;

                for (int c = 0; c < context->channels; c++) {
                    context->dst_im[(n_y_d * context->dst_width + n_x_d) * context->channels + c] =
                            (T) ((1 - lf_weight_y) * (1 - lf_weight_x) *
                                 context->src_im[(n_y_s * context->src_width + n_x_s) * context->channels + c] +
                                 (1 - lf_weight_y) * lf_weight_x *
                                 context->src_im[(n_y_s * context->src_width + n_x_s + 1) * context->channels + c] +
                                 lf_weight_y * (1 - lf_weight_x) *
                                 context->src_im[((n_y_s + 1) * context->src_width + n_x_s) * context->channels + c] +
                                 lf_weight_y * lf_weight_x *
                                 context->src_im[((n_y_s + 1) * context->src_width + n_x_s + 1) * context->channels +
                                                 c]);
                }//end for c
            }
        }


        template<typename T>
        static inline void Resize2d_ResizeImageLinear(const T *src_im, int src_width, int src_height, int channels,
                                                      T *dst_im, int dst_width, int dst_height) {
            if (src_width == dst_width && src_height == dst_height) {
                std::memcpy(dst_im, src_im, src_width * src_height * channels * sizeof(T));
                // memcpy(dst_im, MemoryDevice(CPU), src_width * src_height * channels * sizeof(T),
                //        src_im, MemoryDevice(CPU), src_width * src_height * channels * sizeof(T));
                return;
            }

            double lfx_scl = double(src_width) / dst_width;
            double lfy_scl = double(src_height) / dst_height;
            double bias_x = lfx_scl / 2 - 0.5;
            double bias_y = lfy_scl / 2 - 0.5;

            ResizeImageLinear_kernel_context<T> context{
                    dst_width,
                    lfx_scl,
                    lfy_scl,
                    bias_x,
                    bias_y,
                    src_width,
                    src_height,
                    channels,
                    dst_im,
                    src_im
            };
            auto ctx = ctx::get<RuntimeContext>();
            auto pool = ctx->get_xnn_threadpool();
            pthreadpool_parallelize_1d(pool, (pthreadpool_task_1d_t) ResizeImageLinear_kernel<T>, (void **) &context,
                                       dst_height, 0);
        }


        template<typename T>
        struct ResizeImageCubic_kernel_context {
            double scale_x;
            double scale_y;
            int src_width;
            int src_height;
            int dst_width;
            int dst_height;
            int channels;
            T *dst_im;
            const T *src_im;
            int dstrows;
            int srcrows;
        };

        template<typename T>
        static inline void ResizeImageCubic_kernel(struct ResizeImageCubic_kernel_context<T> *context, size_t j) {
            double fy = (double) ((j + 0.5) * context->scale_y - 0.5);
            int sy = int(floor(fy));
            fy -= sy;
            //sy = std::min(sy, src_height - 3);
            //sy = std::max(1, sy);
            if (sy < 1) {
                fy = 0;
                sy = 1;
            }

            if (sy >= context->src_height - 3) {
                fy = 0, sy = context->src_height - 3;
            }

            const double A = -0.75f;

            double coeffsY[4];
            coeffsY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
            coeffsY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
            coeffsY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
            coeffsY[3] = 1.f - coeffsY[0] - coeffsY[1] - coeffsY[2];

            for (int i = 0; i < context->dst_width; ++i) {
                double fx = (double) ((i + 0.5) * context->scale_x - 0.5);
                int sx = int(floor(fx));
                fx -= sx;

                if (sx < 1) {
                    fx = 0, sx = 1;
                }
                if (sx >= context->src_width - 3) {
                    fx = 0, sx = context->src_width - 3;
                }

                double coeffsX[4];
                coeffsX[0] = ((A * (fx + 1) - 5 * A) * (fx + 1) + 8 * A) * (fx + 1) - 4 * A;
                coeffsX[1] = ((A + 2) * fx - (A + 3)) * fx * fx + 1;
                coeffsX[2] = ((A + 2) * (1 - fx) - (A + 3)) * (1 - fx) * (1 - fx) + 1;
                coeffsX[3] = 1.f - coeffsX[0] - coeffsX[1] - coeffsX[2];

                for (int k = 0; k < context->channels; ++k) {
                    context->dst_im[j * context->dstrows + i * context->channels + k] = (T) ((
                            context->src_im[(sy - 1) * context->srcrows + (sx - 1) * context->channels + k] *
                            coeffsX[0] * coeffsY[0] +
                            context->src_im[(sy) * context->srcrows + (sx - 1) * context->channels + k] * coeffsX[0] *
                            coeffsY[1] +
                            context->src_im[(sy + 1) * context->srcrows + (sx - 1) * context->channels + k] *
                            coeffsX[0] * coeffsY[2] +
                            context->src_im[(sy + 2) * context->srcrows + (sx - 1) * context->channels + k] *
                            coeffsX[0] * coeffsY[3] +

                            context->src_im[(sy - 1) * context->srcrows + (sx) * context->channels + k] * coeffsX[1] *
                            coeffsY[0] +
                            context->src_im[(sy) * context->srcrows + (sx) * context->channels + k] * coeffsX[1] *
                            coeffsY[1] +
                            context->src_im[(sy + 1) * context->srcrows + (sx) * context->channels + k] * coeffsX[1] *
                            coeffsY[2] +
                            context->src_im[(sy + 2) * context->srcrows + (sx) * context->channels + k] * coeffsX[1] *
                            coeffsY[3] +

                            context->src_im[(sy - 1) * context->srcrows + (sx + 1) * context->channels + k] *
                            coeffsX[2] * coeffsY[0] +
                            context->src_im[(sy) * context->srcrows + (sx + 1) * context->channels + k] * coeffsX[2] *
                            coeffsY[1] +
                            context->src_im[(sy + 1) * context->srcrows + (sx + 1) * context->channels + k] *
                            coeffsX[2] * coeffsY[2] +
                            context->src_im[(sy + 2) * context->srcrows + (sx + 1) * context->channels + k] *
                            coeffsX[2] * coeffsY[3] +

                            context->src_im[(sy - 1) * context->srcrows + (sx + 2) * context->channels + k] *
                            coeffsX[3] * coeffsY[0] +
                            context->src_im[(sy) * context->srcrows + (sx + 2) * context->channels + k] * coeffsX[3] *
                            coeffsY[1] +
                            context->src_im[(sy + 1) * context->srcrows + (sx + 2) * context->channels + k] *
                            coeffsX[3] * coeffsY[2] +
                            context->src_im[(sy + 2) * context->srcrows + (sx + 2) * context->channels + k] *
                            coeffsX[3] * coeffsY[3]));

                }//end k
            }
        }

        template<typename T>
        static inline void Resize2d_ResizeImageCubic(const T *src_im, int src_width, int src_height, int channels,
                                                     T *dst_im, int dst_width, int dst_height) {
            double scale_x = (double) src_width / dst_width;
            double scale_y = (double) src_height / dst_height;

            int srcrows = src_width * channels;
            int dstrows = dst_width * channels;

            ResizeImageCubic_kernel_context<T> context{
                    scale_x, scale_y,
                    src_width, src_height,
                    dst_width, dst_height,
                    channels,
                    dst_im, src_im,
                    dstrows,
                    srcrows
            };
            auto ctx = ctx::get<RuntimeContext>();
            auto pool = ctx->get_xnn_threadpool();
            pthreadpool_parallelize_1d(pool, (pthreadpool_task_1d_t) ResizeImageCubic_kernel<T>,
                                       (void **) &context, dst_height, 0);
        }


        template<typename T>
        struct ResizeHard_kernel_context {
            int dst_width;
            double lfx_scl;
            double lfy_scl;
            int src_width;
            int src_height;
            int channels;
            T *dst_im;
            const T *src_im;
        };

        template<typename T>
        static inline void ResizeHard_kernel(struct ResizeHard_kernel_context<T> *context, size_t n_y_d) {
            for (int n_x_d = 0; n_x_d < context->dst_width; n_x_d++) {
                float lf_x_s = context->lfx_scl * n_x_d;
                float lf_y_s = context->lfy_scl * n_y_d;

                auto n_x_s = int(lf_x_s);
                auto n_y_s = int(lf_y_s);

                n_x_s = n_x_s >= 0 ? n_x_s : 0;
                n_x_s = n_x_s < context->src_width - 1 ? n_x_s : context->src_width - 1;
                n_y_s = n_y_s >= 0 ? n_y_s : 0;
                n_y_s = n_y_s < context->src_height - 1 ? n_y_s : context->src_height - 1;

                for (int c = 0; c < context->channels; c++) {
                    context->dst_im[(n_y_d * context->dst_width + n_x_d) * context->channels + c] = context->src_im[
                            (n_y_s * context->src_width + n_x_s) * context->channels + c];
                }//end for c
            }
        }

        template<typename T>
        static inline void Resize2d_ResizeHard(const T *src_im, int src_width, int src_height, int channels,
                                               T *dst_im, int dst_width, int dst_height) {
            if (src_width == dst_width && src_height == dst_height) {
                std::memcpy(dst_im, src_im, src_width * src_height * channels * sizeof(T));
                // memcpy(dst_im, MemoryDevice(CPU), src_width * src_height * channels * sizeof(T),
                //        src_im, MemoryDevice(CPU), src_width * src_height * channels * sizeof(T));
                return;
            }

            const float lfx_scl = float(src_width) / dst_width;
            const float lfy_scl = float(src_height) / dst_height;

            ResizeHard_kernel_context<T> context{
                    dst_width,
                    lfx_scl,
                    lfy_scl,
                    src_width,
                    src_height,
                    channels,
                    dst_im, src_im
            };
            auto ctx = ctx::get<RuntimeContext>();
            auto pool = ctx->get_xnn_threadpool();
            pthreadpool_parallelize_1d(pool, (pthreadpool_task_1d_t) ResizeHard_kernel<T>, (void **) &context,
                                       dst_height, 0);
        }


        template<typename T>
        struct ResizeNearest_kernel_context {
            int dst_width;
            double lfx_scl;
            double lfy_scl;
            double bias_x;
            double bias_y;
            int src_width;
            int src_height;
            int channels;
            T *dst_im;
            const T *src_im;
        };

        template<typename T>
        static inline void ResizeNearest_kernel(struct ResizeNearest_kernel_context<T> *context, size_t n_y_d) {
            for (int n_x_d = 0; n_x_d < context->dst_width; n_x_d++) {
                double lf_x_s = context->lfx_scl * n_x_d + context->bias_x;
                double lf_y_s = context->lfy_scl * n_y_d + context->bias_y;

                auto n_x_s = int(std::round(lf_x_s));
                auto n_y_s = int(std::round(lf_y_s));

                n_x_s = n_x_s >= 0 ? n_x_s : 0;
                n_x_s = n_x_s < context->src_width - 1 ? n_x_s : context->src_width - 1;
                n_y_s = n_y_s >= 0 ? n_y_s : 0;
                n_y_s = n_y_s < context->src_height - 1 ? n_y_s : context->src_height - 1;

                for (int c = 0; c < context->channels; c++) {
                    context->dst_im[(n_y_d * context->dst_width + n_x_d) * context->channels + c] = context->src_im[
                            (n_y_s * context->src_width + n_x_s) * context->channels + c];
                }//end for c
            }
        }

        template<typename T>
        static inline void Resize2d_ResizeNearest(const T *src_im, int src_width, int src_height, int channels,
                                                  T *dst_im, int dst_width, int dst_height) {
            if (src_width == dst_width && src_height == dst_height) {
                std::memcpy(dst_im, src_im, src_width * src_height * channels * sizeof(T));
                // memcpy(dst_im, MemoryDevice(CPU), src_width * src_height * channels * sizeof(T),
                //        src_im, MemoryDevice(CPU), src_width * src_height * channels * sizeof(T));
                return;
            }

            double lfx_scl = double(src_width) / dst_width;
            double lfy_scl = double(src_height) / dst_height;
            double bias_x = lfx_scl / 2 - 0.5;
            double bias_y = lfy_scl / 2 - 0.5;

            ResizeNearest_kernel_context<T> context{
                    dst_width,
                    lfx_scl,
                    lfy_scl,
                    bias_x,
                    bias_y,
                    src_width,
                    src_height,
                    channels,
                    dst_im, src_im
            };
            auto ctx = ctx::get<RuntimeContext>();
            auto pool = ctx->get_xnn_threadpool();
            pthreadpool_parallelize_1d(pool, (pthreadpool_task_1d_t) ResizeNearest_kernel<T>, (void **) &context,
                                       dst_height, 0);
        }


        template<typename T>
        static inline void resize_linear(const Tensor *x, Tensor *y, int x_height, int x_width,
                                         int y_height, int y_width, unsigned int x_offset, unsigned int y_offset,
                                         int channels) {

            const T *psrc = x->data<T>() + x_offset;
            T *pdst = y->data<T>() + y_offset;
            Resize2d_ResizeImageLinear<T>(psrc, x_width, x_height, channels, pdst, y_width, y_height);
        }


        template<typename T>
        static inline void resize_cubic(const Tensor *x, Tensor *y, int x_height, int x_width,
                                        int y_height, int y_width, unsigned int x_offset, unsigned int y_offset,
                                        int channels) {

            const T *psrc = x->data<T>() + x_offset;
            T *pdst = y->data<T>() + y_offset;
            Resize2d_ResizeImageCubic<T>(psrc, x_width, x_height, channels, pdst, y_width, y_height);
        }


        template<typename T>
        static inline void resize_nearest(const Tensor *x, Tensor *y, int x_height, int x_width,
                                          int y_height, int y_width, unsigned int x_offset, unsigned int y_offset,
                                          int channels) {

            const T *psrc = x->data<T>() + x_offset;
            T *pdst = y->data<T>() + y_offset;
            Resize2d_ResizeNearest<T>(psrc, x_width, x_height, channels, pdst, y_width, y_height);
        }


        template<typename T>
        static inline void resize_hard(const Tensor *x, Tensor *y, int x_height, int x_width,
                                       int y_height, int y_width, unsigned int x_offset, unsigned int y_offset,
                                       int channels) {

            const T *psrc = x->data<T>() + x_offset;
            T *pdst = y->data<T>() + y_offset;
            Resize2d_ResizeHard<T>(psrc, x_width, x_height, channels, pdst, y_width, y_height);
        }

        template<typename T>
        static inline void batch_resize_linear(int number, const Tensor *x, Tensor *y, int x_height, int x_width,
                                               int y_height, int y_width,
                                               unsigned int x_batch_step, unsigned int y_batch_step,
                                               int channels) {
            for (int k = 0; k < number; k++) {
                resize_linear<T>(x, y, x_height, x_width, y_height, y_width,
                                 k * x_batch_step, k * y_batch_step, channels);
            }
        }

        template<typename T>
        static inline void batch_resize_cubic(int number, const Tensor *x, Tensor *y, int x_height, int x_width,
                                              int y_height, int y_width,
                                              unsigned int x_batch_step, unsigned int y_batch_step,
                                              int channels) {
            for (int k = 0; k < number; k++) {
                resize_cubic<T>(x, y, x_height, x_width, y_height, y_width,
                                k * x_batch_step, k * y_batch_step, channels);
            }
        }

        template<typename T>
        static inline void batch_resize_nearest(int number, const Tensor *x, Tensor *y, int x_height, int x_width,
                                                int y_height, int y_width,
                                                unsigned int x_batch_step, unsigned int y_batch_step,
                                                int channels) {
            for (int k = 0; k < number; k++) {
                resize_nearest<T>(x, y, x_height, x_width, y_height, y_width,
                                  k * x_batch_step, k * y_batch_step, channels);
            }
        }

        template<typename T>
        static inline void batch_resize_hard(int number, const Tensor *x, Tensor *y, int x_height, int x_width,
                                             int y_height, int y_width,
                                             unsigned int x_batch_step, unsigned int y_batch_step,
                                             int channels) {
            for (int k = 0; k < number; k++) {
                resize_hard<T>(x, y, x_height, x_width, y_height, y_width,
                               k * x_batch_step, k * y_batch_step, channels);
            }
        }

        template<typename T>
        static void batch_resize(int number, const Tensor *x, Tensor *y, int x_height, int x_width,
                                 int y_height, int y_width,
                                 unsigned int x_batch_step, unsigned int y_batch_step,
                                 int channels, Resize2DType type) {
            if (type == Resize2DType::LINEAR) {
                batch_resize_linear<T>(number, x, y,
                                       x_height, x_width,
                                       y_height, y_width,
                                       x_batch_step, y_batch_step, channels);

            } else if (type == Resize2DType::CUBIC) {
                batch_resize_cubic<T>(number, x, y,
                                      x_height, x_width,
                                      y_height, y_width,
                                      x_batch_step, y_batch_step, channels);
            } else if (type == Resize2DType::HARD) {
                batch_resize_hard<T>(number, x, y,
                                     x_height, x_width,
                                     y_height, y_width,
                                     x_batch_step, y_batch_step, channels);
            } else {
                batch_resize_nearest<T>(number, x, y,
                                        x_height, x_width,
                                        y_height, y_width,
                                        x_batch_step, y_batch_step, channels);

            }
        }

        void Resize2D::resize2d(const Tensor &x, int i, Resize2DType type, Tensor &out) {
            auto &output_shape = out.sizes();

            int y_height = out.size(i);
            int y_width = out.size(i + 1);
            int x_height = x.size(i);
            int x_width = x.size(i + 1);

            int number, channels;
            number = channels = 1;

            for (int k = 0; k < i; k++) {
                number *= output_shape[k];
            }

            for (int k = i + 2; k < output_shape.size(); k++) {
                channels *= output_shape[k];
            }

            int y_batch_step = channels * y_height * y_width;
            int x_batch_step = channels * x_height * x_width;

            const Tensor *input = &x;
            Tensor *output = &out;
            ts::DTYPE dtype = output->dtype();

            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { batch_resize<TYPE>( \
                        number, input, output, \
                        x_height, x_width, \
                        y_height, y_width, \
                        x_batch_step, y_batch_step, channels, type); break; }
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
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype)
                                 << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace xnn;
TS_REGISTER_OPERATOR(Resize2D, ts::XNNPACK, name::layer::resize2d())
