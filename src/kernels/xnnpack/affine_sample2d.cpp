#include <kernels/cpu/affine_sample2d.h>
#include <core/tensor_builder.h>

#include <global/operator_factory.h>
#include <backend/name.h>
#include <utils/assert.h>
#include <core/device.h>
#include <vector>
#include <algorithm>
#include "affine_sample2d.h"
#include "runtime/runtime.h"


namespace ts {
    namespace xnn {
        void Affine_Sample2D::init() {
            supper::init();

            auto ctx = ctx::get<RuntimeContext>();
            m_pool = ctx->get_xnn_threadpool();
        }

        template<typename T>
        struct vec3d {
        public:
            vec3d() = default;

            template<typename X, typename Y, typename Z>
            vec3d(X x, Y y, Z z): x(T(x)), y(T(y)), z(T(z)) {}

            T x;
            T y;
            T z;
        };

        template<typename T>
        static vec3d<T> transform(float rz00, float rz01, float rz02, float rz10, float rz11, float rz12,
                                  float rz20, float rz21, float rz22, const vec3d<T> &pos) {

            vec3d<T> ret;
            ret.x = rz00 * pos.x + rz01 * pos.y + rz02 * pos.z;
            ret.y = rz10 * pos.x + rz11 * pos.y + rz12 * pos.z;
            ret.z = rz20 * pos.x + rz21 * pos.y + rz22 * pos.z;
            return ret;
        }

        template<typename TO, typename FROM>
        static TO clamp(FROM from) {
            constexpr auto MAX = FROM(std::numeric_limits<TO>::max());
            constexpr auto MIN = FROM(std::numeric_limits<TO>::lowest());
            return TO(std::max(MIN, std::min(MAX, from)));
        }

        template<typename T>
        struct linear_kernel_context {
            int dst_width;
            float rz00;
            float rz01;
            float rz02;
            float rz10;
            float rz11;
            float rz12;
            float rz20;
            float rz21;
            float rz22;
            int src_height;
            int src_width;
            base::AffineOuterMode outer_mode;
            int channels;
            T outer_value;
            T *dst_im;
            const T *src_im;
        };

        template<typename T>
        static inline void linear_kernel(struct linear_kernel_context<T> *context, size_t n_y_d) {
            for (int n_x_d = 0; n_x_d < context->dst_width; n_x_d++) {
                vec3d<float> cur(n_x_d, n_y_d, 1);
                auto location = transform<float>(context->rz00, context->rz01, context->rz02,
                                                 context->rz10, context->rz11, context->rz12,
                                                 context->rz20, context->rz21, context->rz22, cur);

                double lf_x_s = location.x;
                double lf_y_s = location.y;

                auto inner = lf_x_s >= 0 && lf_x_s < context->src_width - 1 &&
                             lf_y_s >= 0 && lf_y_s < context->src_height - 1;

                if (!inner && context->outer_mode == base::AffineOuterMode::VALUE) {
                    for (int c = 0; c < context->channels; c++) {
                        context->dst_im[(n_y_d * context->dst_width + n_x_d) * context->channels +
                                        c] = context->outer_value;
                    }
                    continue;
                }

                lf_x_s = lf_x_s >= 0 ? lf_x_s : 0;
                lf_x_s = lf_x_s < context->src_width - 1 ? lf_x_s : context->src_width - 1 - 1e-5;
                lf_y_s = lf_y_s >= 0 ? lf_y_s : 0;
                lf_y_s = lf_y_s < context->src_height - 1 ? lf_y_s : context->src_height - 1 - 1e-5;

                int n_x_s = int(lf_x_s);
                int n_y_s = int(lf_y_s);

                double lf_weight_x = lf_x_s - n_x_s;
                double lf_weight_y = lf_y_s - n_y_s;

                for (int c = 0; c < context->channels; c++) {
                    context->dst_im[(n_y_d * context->dst_width + n_x_d) * context->channels + c] = clamp<T, double>(
                            ((1 - lf_weight_y) * (1 - lf_weight_x) *
                             context->src_im[(n_y_s * context->src_width + n_x_s) * context->channels + c] +
                             (1 - lf_weight_y) * lf_weight_x *
                             context->src_im[(n_y_s * context->src_width + n_x_s + 1) * context->channels + c] +
                             lf_weight_y * (1 - lf_weight_x) *
                             context->src_im[((n_y_s + 1) * context->src_width + n_x_s) * context->channels + c] +
                             lf_weight_y * lf_weight_x *
                             context->src_im[((n_y_s + 1) * context->src_width + n_x_s + 1) * context->channels + c]));

                } //end for c
            }
        }

        template<typename T>
        static void affine_sample2d_linear(const Tensor *x, Tensor *y, int src_height, int src_width,
                                           int dst_height, int dst_width,
                                           unsigned int x_offset, unsigned int y_offset,
                                           int channels, float rz00, float rz01, float rz02, float rz10,
                                           float rz11, float rz12, float rz20, float rz21, float rz22,
                                           pthreadpool_t pool,
                                           base::AffineOuterMode outer_mode = base::AffineOuterMode::NEAREST,
                                           T outer_value = T(0)) {

            const T *src_im = x->data<T>() + x_offset;
            T *dst_im = y->data<T>() + y_offset;

            linear_kernel_context<T> context{
                    dst_width,
                    rz00, rz01, rz02, rz10,
                    rz11, rz12, rz20, rz21, rz22,
                    src_height, src_width,
                    outer_mode,
                    channels, outer_value,
                    dst_im, src_im,
            };
            pthreadpool_parallelize_1d(pool, (pthreadpool_task_1d_t) linear_kernel<T>,
                                       (void **) &context, dst_height, 0);
        }

        template<typename T>
        struct nearest_kernel_context {
            int dst_width;
            float rz00;
            float rz01;
            float rz02;
            float rz10;
            float rz11;
            float rz12;
            float rz20;
            float rz21;
            float rz22;
            int src_height;
            int src_width;
            base::AffineOuterMode outer_mode;
            int channels;
            T outer_value;
            T *dst_im;
            const T *src_im;
        };

        template<typename T>
        static inline void nearest_kernel(struct nearest_kernel_context<T> *context, size_t n_y_d) {
            for (int n_x_d = 0; n_x_d < context->dst_width; n_x_d++) {

                vec3d<float> cur(n_x_d, n_y_d, 1);
                auto location = transform<float>(context->rz00, context->rz01, context->rz02,
                                                 context->rz10, context->rz11, context->rz12,
                                                 context->rz20, context->rz21, context->rz22, cur);

                double lf_x_s = location.x;
                double lf_y_s = location.y;

                auto n_x_s = int(std::round(lf_x_s));
                auto n_y_s = int(std::round(lf_y_s));

                auto inner = n_x_s >= 0 && n_x_s < context->src_width - 1 &&
                             n_y_s >= 0 && n_y_s < context->src_height - 1;

                if (!inner && context->outer_mode == base::AffineOuterMode::VALUE) {
                    for (int c = 0; c < context->channels; c++) {
                        context->dst_im[(n_y_d * context->dst_width + n_x_d) * context->channels +
                                        c] = context->outer_value;
                    }
                    continue;
                }

                n_x_s = n_x_s >= 0 ? n_x_s : 0;
                n_x_s = n_x_s < context->src_width - 1 ? n_x_s : context->src_width - 1;
                n_y_s = n_y_s >= 0 ? n_y_s : 0;
                n_y_s = n_y_s < context->src_height - 1 ? n_y_s : context->src_height - 1;

                for (int c = 0; c < context->channels; c++) {
                    context->dst_im[(n_y_d * context->dst_width + n_x_d) * context->channels + c] =
                            context->src_im[(n_y_s * context->src_width + n_x_s) * context->channels + c];
                }//end for c
            }
        };

        template<typename T>
        static void affine_sample2d_nearest(const Tensor *x, Tensor *y, int src_height, int src_width,
                                            int dst_height, int dst_width,
                                            unsigned int x_offset, unsigned int y_offset,
                                            int channels, float rz00, float rz01, float rz02, float rz10,
                                            float rz11, float rz12, float rz20, float rz21, float rz22,
                                            pthreadpool_t pool,
                                            base::AffineOuterMode outer_mode = base::AffineOuterMode::NEAREST,
                                            T outer_value = T(0)) {
            const T *src_im = x->data<T>() + x_offset;
            T *dst_im = y->data<T>() + y_offset;

            nearest_kernel_context<T> context{
                    dst_width,
                    rz00, rz01, rz02, rz10,
                    rz11, rz12, rz20, rz21, rz22,
                    src_height, src_width,
                    outer_mode,
                    channels, outer_value,
                    dst_im, src_im
            };
            pthreadpool_parallelize_1d(pool, (pthreadpool_task_1d_t) nearest_kernel<T>,
                                       (void **) &context, dst_height, 0);
        }

        template<typename T>
        struct hard_kernel_context {
            int dst_width;
            float rz00;
            float rz01;
            float rz02;
            float rz10;
            float rz11;
            float rz12;
            float rz20;
            float rz21;
            float rz22;
            int src_height;
            int src_width;
            T *dst_im;
            const T *src_im;
            base::AffineOuterMode outer_mode;
            int channels;
            T outer_value;
        };

        template<typename T>
        static void hard_kernel(struct hard_kernel_context<T> *context, size_t n_y_d) {
            for (int n_x_d = 0; n_x_d < context->dst_width; n_x_d++) {

                vec3d<float> cur(n_x_d, n_y_d, 1);
                auto location = transform<float>(context->rz00, context->rz01, context->rz02,
                                                 context->rz10, context->rz11, context->rz12,
                                                 context->rz20, context->rz21, context->rz22, cur);

                double lf_x_s = location.x;
                double lf_y_s = location.y;

                auto n_x_s = int(lf_x_s);
                auto n_y_s = int(lf_y_s);

                auto inner = n_x_s >= 0 && n_x_s < context->src_width - 1 &&
                             n_y_s >= 0 && n_y_s < context->src_height - 1;

                if (!inner && context->outer_mode == base::AffineOuterMode::VALUE) {
                    for (int c = 0; c < context->channels; c++) {
                        context->dst_im[(n_y_d * context->dst_width + n_x_d) * context->channels +
                                        c] = context->outer_value;
                    }
                    continue;
                }

                n_x_s = n_x_s >= 0 ? n_x_s : 0;
                n_x_s = n_x_s < context->src_width - 1 ? n_x_s : context->src_width - 1;
                n_y_s = n_y_s >= 0 ? n_y_s : 0;
                n_y_s = n_y_s < context->src_height - 1 ? n_y_s : context->src_height - 1;

                for (int c = 0; c < context->channels; c++) {
                    context->dst_im[(n_y_d * context->dst_width + n_x_d) * context->channels + c] =
                            context->src_im[(n_y_s * context->src_width + n_x_s) * context->channels + c];
                }//end for c
            }
        }


        template<typename T>
        static void affine_sample2d_hard(const Tensor *x, Tensor *y, int src_height, int src_width,
                                         int dst_height, int dst_width,
                                         unsigned int x_offset, unsigned int y_offset,
                                         int channels, float rz00, float rz01, float rz02, float rz10,
                                         float rz11, float rz12, float rz20, float rz21, float rz22,
                                         pthreadpool_t pool,
                                         base::AffineOuterMode outer_mode = base::AffineOuterMode::NEAREST,
                                         T outer_value = T(0)) {
            const T *src_im = x->data<T>() + x_offset;
            T *dst_im = y->data<T>() + y_offset;

            hard_kernel_context<T> context{
                    dst_width,
                    rz00, rz01, rz02, rz10,
                    rz11, rz12, rz20, rz21, rz22,
                    src_height, src_width,
                    dst_im, src_im,
                    outer_mode,
                    channels, outer_value
            };
            pthreadpool_parallelize_1d(pool, (pthreadpool_task_1d_t) hard_kernel<T>,
                                       (void **) &context, dst_height, 0);
        }

        template<typename T>
        struct cubic_kernel_context {
            int y_width;
            float rz00;
            float rz01;
            float rz02;
            float rz10;
            float rz11;
            float rz12;
            float rz20;
            float rz21;
            float rz22;
            int x_width;
            int x_height;
            const double A;
            T *pdst;
            const T *psrc;
            T outer_value;
            const int dstrows;
            const int srcrows;
            int channels;
            base::AffineOuterMode outer_mode;
        };

        template<typename T>
        static inline void cubic_kernel(struct cubic_kernel_context<T> *context, size_t m) {
            double coeffsY[4];
            double coeffsX[4];
            for (int n = 0; n < context->y_width; n++) {
                vec3d<float> cur(n, m, 1);
                auto location = transform<float>(context->rz00, context->rz01, context->rz02,
                                                 context->rz10, context->rz11, context->rz12,
                                                 context->rz20, context->rz21, context->rz22, cur);

                double fy = location.y;
                auto sy = int(std::floor(fy));
                fy -= sy;

                double fx = location.x;
                auto sx = int(std::floor(fx));
                fx -= sx;

                auto outter = sy < 1 || sy >= context->x_height - 3 || sx < 1 || sx >= context->x_width - 3;

                if (outter && context->outer_mode == base::AffineOuterMode::VALUE) {
                    for (int k = 0; k < context->channels; k++) {
                        context->pdst[m * context->dstrows + n * context->channels + k] = context->outer_value;
                    }
                    //continue;
                } else {
                    if (sy < 1) {
                        fy = 0;
                        sy = 1;
                    }
                    if (sy >= context->x_height - 3) {
                        fy = 0;
                        sy = context->x_height - 3;
                    }
                    if (sx < 1) {
                        fx = 0;
                        sx = 1;
                    }
                    if (sx >= context->x_width - 3) {
                        fx = 0;
                        sx = context->x_width - 3;
                    }

                    coeffsY[0] = ((context->A * (fy + 1) - 5 * context->A) * (fy + 1) + 8 * context->A) * (fy + 1) -
                                 4 * context->A;
                    coeffsY[1] = ((context->A + 2) * fy - (context->A + 3)) * fy * fy + 1;
                    coeffsY[2] = ((context->A + 2) * (1 - fy) - (context->A + 3)) * (1 - fy) * (1 - fy) + 1;
                    coeffsY[3] = 1.f - coeffsY[0] - coeffsY[1] - coeffsY[2];

                    coeffsX[0] = ((context->A * (fx + 1) - 5 * context->A) * (fx + 1) + 8 * context->A) * (fx + 1) -
                                 4 * context->A;
                    coeffsX[1] = ((context->A + 2) * fx - (context->A + 3)) * fx * fx + 1;
                    coeffsX[2] = ((context->A + 2) * (1 - fx) - (context->A + 3)) * (1 - fx) * (1 - fx) + 1;
                    coeffsX[3] = 1.f - coeffsX[0] - coeffsX[1] - coeffsX[2];

                    for (int k = 0; k < context->channels; k++) {
                        context->pdst[m * context->dstrows + n * context->channels + k] = clamp<T, double>(((
                                context->psrc[(sy - 1) * context->srcrows + (sx - 1) * context->channels + k] *
                                coeffsX[0] * coeffsY[0] +
                                context->psrc[(sy) * context->srcrows + (sx - 1) * context->channels + k] * coeffsX[0] *
                                coeffsY[1] +
                                context->psrc[(sy + 1) * context->srcrows + (sx - 1) * context->channels + k] *
                                coeffsX[0] * coeffsY[2] +
                                context->psrc[(sy + 2) * context->srcrows + (sx - 1) * context->channels + k] *
                                coeffsX[0] * coeffsY[3] +

                                context->psrc[(sy - 1) * context->srcrows + (sx) * context->channels + k] * coeffsX[1] *
                                coeffsY[0] +
                                context->psrc[(sy) * context->srcrows + (sx) * context->channels + k] * coeffsX[1] *
                                coeffsY[1] +
                                context->psrc[(sy + 1) * context->srcrows + (sx) * context->channels + k] * coeffsX[1] *
                                coeffsY[2] +
                                context->psrc[(sy + 2) * context->srcrows + (sx) * context->channels + k] * coeffsX[1] *
                                coeffsY[3] +

                                context->psrc[(sy - 1) * context->srcrows + (sx + 1) * context->channels + k] *
                                coeffsX[2] * coeffsY[0] +
                                context->psrc[(sy) * context->srcrows + (sx + 1) * context->channels + k] * coeffsX[2] *
                                coeffsY[1] +
                                context->psrc[(sy + 1) * context->srcrows + (sx + 1) * context->channels + k] *
                                coeffsX[2] * coeffsY[2] +
                                context->psrc[(sy + 2) * context->srcrows + (sx + 1) * context->channels + k] *
                                coeffsX[2] * coeffsY[3] +

                                context->psrc[(sy - 1) * context->srcrows + (sx + 2) * context->channels + k] *
                                coeffsX[3] * coeffsY[0] +
                                context->psrc[(sy) * context->srcrows + (sx + 2) * context->channels + k] * coeffsX[3] *
                                coeffsY[1] +
                                context->psrc[(sy + 1) * context->srcrows + (sx + 2) * context->channels + k] *
                                coeffsX[3] * coeffsY[2] +
                                context->psrc[(sy + 2) * context->srcrows + (sx + 2) * context->channels + k] *
                                coeffsX[3] * coeffsY[3])));

                    }
                }
            }
        }

        template<typename T>
        static void affine_sample2d_cubic(const Tensor *x, Tensor *y, int x_height, int x_width,
                                          int y_height, int y_width,
                                          unsigned int x_offset, unsigned int y_offset,
                                          int channels, float rz00, float rz01, float rz02, float rz10,
                                          float rz11, float rz12, float rz20, float rz21, float rz22,
                                          pthreadpool_t pool,
                                          base::AffineOuterMode outer_mode = base::AffineOuterMode::NEAREST,
                                          T outer_value = T(0)) {
            const T *psrc = x->data<T>() + x_offset;
            T *pdst = y->data<T>() + y_offset;
            const double A = -0.75f;
            const int srcrows = x_width * channels;
            const int dstrows = y_width * channels;

            cubic_kernel_context<T> kernel_context{
                    y_width,
                    rz00, rz01, rz02, rz10,
                    rz11, rz12, rz20, rz21, rz22,
                    x_width, x_height,
                    A,
                    pdst, psrc, outer_value,
                    dstrows, srcrows,
                    channels,
                    outer_mode
            };

            pthreadpool_parallelize_1d(pool, (pthreadpool_task_1d_t) cubic_kernel<T>,
                                       (void **) &kernel_context, y_height, 0);
        }


        template<typename T>
        static void batch_affine_sample2d(int number, const Tensor *x, Tensor *y, int x_height, int x_width,
                                          int y_height, int y_width,
                                          unsigned int x_batch_step, unsigned int y_batch_step,
                                          int channels, Affine_Sample2DType type, float rz00, float rz01, float rz02,
                                          float rz10,
                                          float rz11, float rz12, float rz20, float rz21, float rz22,
                                          pthreadpool_t pool, base::AffineOuterMode outer_mode, T outer_value = T(0)) {
            if (type == Affine_Sample2DType::CUBIC) {
                for (int k = 0; k < number; k++) {
                    affine_sample2d_cubic<T>(x, y, x_height, x_width, y_height, y_width, k * x_batch_step,
                                             k * y_batch_step, channels,
                                             rz00, rz01, rz02, rz10, rz11, rz12, rz20, rz21, rz22, pool, outer_mode,
                                             T(outer_value));
                }
            } else if (type == Affine_Sample2DType::NEAREST) {

                for (int k = 0; k < number; k++) {
                    affine_sample2d_nearest<T>(x, y, x_height, x_width, y_height, y_width, k * x_batch_step,
                                               k * y_batch_step, channels,
                                               rz00, rz01, rz02, rz10, rz11, rz12, rz20, rz21, rz22, pool, outer_mode,
                                               T(outer_value));
                }
            } else if (type == Affine_Sample2DType::HARD) {

                for (int k = 0; k < number; k++) {
                    affine_sample2d_hard<T>(x, y, x_height, x_width, y_height, y_width, k * x_batch_step,
                                            k * y_batch_step, channels,
                                            rz00, rz01, rz02, rz10, rz11, rz12, rz20, rz21, rz22, pool, outer_mode,
                                            T(outer_value));
                }
            } else { //LINEAR
                for (int k = 0; k < number; k++) {
                    affine_sample2d_linear<T>(x, y, x_height, x_width, y_height, y_width, k * x_batch_step,
                                              k * y_batch_step, channels,
                                              rz00, rz01, rz02, rz10, rz11, rz12, rz20, rz21, rz22, pool, outer_mode,
                                              T(outer_value));
                }
            }
        }

        void Affine_Sample2D::affine_sample_run(const Tensor &x, float rz00, float rz01, float rz02, float rz10,
                                                float rz11, float rz12, float rz20, float rz21, float rz22,
                                                Affine_Sample2DType type, int dim,
                                                base::AffineOuterMode outer_mode, float outer_value,
                                                Tensor &out) {

            auto &output_shape = out.sizes();

            int y_height = out.size(dim);
            int y_width = out.size(dim + 1);
            int x_height = x.size(dim);

            // auto &input_shape = x.sizes();

            int x_width = x.size(dim + 1);

            int number, channels;
            number = channels = 1;

            for (int k = 0; k < dim; k++) {
                number *= output_shape[k];
            }

            for (int k = dim + 2; k < output_shape.size(); k++) {
                channels *= output_shape[k];
            }

            int y_batch_step = channels * y_height * y_width;
            int x_batch_step = channels * x_height * x_width;

            const Tensor *input = &x;
            Tensor *output = &out;
            ts::DTYPE dtype = output->dtype();

            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { batch_affine_sample2d<TYPE>( \
                        number, input, output, \
                        x_height, x_width, \
                        y_height, y_width, \
                        x_batch_step, y_batch_step, channels, type, rz00,rz01,rz02,rz10,rz11,rz12,rz20,rz21,rz22, \
                        m_pool, outer_mode, TYPE(outer_value)); break; }
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
TS_REGISTER_OPERATOR(Affine_Sample2D, ts::XNNPACK, name::layer::affine_sample2d())
