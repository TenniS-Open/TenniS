#include <kernels/cpu/affine_sample2d.h>
#include <core/tensor_builder.h>

#include <global/operator_factory.h>
#include <backend/name.h>
#include <utils/assert.h>
#include <core/device.h>
#include <vector>


namespace ts {
    namespace cpu {

        template <typename T>
        struct vec3d {
            T x;
            T y;
            T z; 
        };

        template <typename T>
        static vec3d<T> transform(float rz00, float rz01, float rz02,float rz10, float rz11, float rz12,
                        float rz20, float rz21, float rz22, const vec3d<T> &pos) {

            vec3d<T> ret;
            ret.x = rz00 * pos.x + rz01 * pos.y + rz02 *pos.z;
            ret.y = rz10 * pos.x + rz11 * pos.y + rz12 *pos.z;
            ret.z = rz20 * pos.x + rz21 * pos.y + rz22 *pos.z;
            return ret;
        }

        template<typename T>
        static void affine_sample2d_linear(const Tensor *x, Tensor *y, int src_height, int src_width,
                                 int dst_height, int dst_width,
                                 unsigned int x_offset, unsigned int y_offset,
                                 int channels, float rz00, float rz01, float rz02, float rz10,
                                 float rz11, float rz12, float rz20, float rz21, float rz22,
                                 T outer_value = T(0)) {

            const T *src_im = x->data<T>() + x_offset;
            T *dst_im = y->data<T>() + y_offset;

            for (int n_y_d = 0; n_y_d < dst_height; n_y_d++) {
                for (int n_x_d = 0; n_x_d < dst_width; n_x_d++) {
                    vec3d<float> cur;
                    cur.x = n_x_d;
                    cur.y = n_y_d;
                    cur.z = 1;
                    auto location = transform<float>(rz00,rz01,rz02,rz10,rz11,rz12,rz20,rz21,rz22, cur);

                    double lfx_scl =  location.x / n_x_d;
                    double lfy_scl =  location.y / n_y_d;
                    double bias_x = lfx_scl / 2 - 0.5;
                    double bias_y = lfy_scl / 2 - 0.5;
                    double lf_x_s = lfx_scl * n_x_d + bias_x;
                    double lf_y_s = lfy_scl * n_y_d + bias_y;

                    auto inner = lf_x_s >= 0 && lf_x_s < src_width - 1 &&
                            lf_y_s >= 0 && lf_y_s < src_height - 1;

                    if (!inner) {
                        for (int c = 0; c < channels; c++) {
                            dst_im[(n_y_d * dst_width + n_x_d) * channels + c] = outer_value;
                        }
                        continue;
                    }

                    int n_x_s = int(lf_x_s);
                    int n_y_s = int(lf_y_s);

                    double lf_weight_x = lf_x_s - n_x_s;
                    double lf_weight_y = lf_y_s - n_y_s;

                    for (int c = 0; c < channels; c++) {
                        dst_im[(n_y_d * dst_width + n_x_d) * channels + c] =
                                (T) ((1 - lf_weight_y) * (1 - lf_weight_x) *
                                     src_im[(n_y_s * src_width + n_x_s) * channels + c] +
                                     (1 - lf_weight_y) * lf_weight_x *
                                     src_im[(n_y_s * src_width + n_x_s + 1) * channels + c] +
                                     lf_weight_y * (1 - lf_weight_x) *
                                     src_im[((n_y_s + 1) * src_width + n_x_s) * channels + c] +
                                     lf_weight_y * lf_weight_x *
                                     src_im[((n_y_s + 1) * src_width + n_x_s + 1) * channels + c]);

                    }//end for c
                }
            }
        }

        template<typename T>
        static void affine_sample2d_nearest(const Tensor *x, Tensor *y, int src_height, int src_width,
                                 int dst_height, int dst_width,
                                 unsigned int x_offset, unsigned int y_offset,
                                 int channels, float rz00, float rz01, float rz02, float rz10,
                                 float rz11, float rz12, float rz20, float rz21, float rz22,
                                 T outer_value = T(0)) {
            const T *src_im = x->data<T>() + x_offset;
            T *dst_im = y->data<T>() + y_offset;

            for (int n_y_d = 0; n_y_d < dst_height; n_y_d++) {
                for (int n_x_d = 0; n_x_d < dst_width; n_x_d++) {

                    vec3d<T> cur;
                    cur.x = n_x_d;
                    cur.y = n_y_d;
                    cur.z = 1;
                    vec3d<T> location = transform<T>(rz00,rz01,rz02,rz10,rz11,rz12,rz20,rz21,rz22, cur);

                    double lfx_scl =  location.x / n_x_d;
                    double lfy_scl =  location.y / n_y_d;

                    double bias_x = lfx_scl / 2 - 0.5;
                    double bias_y = lfy_scl / 2 - 0.5;
                    double lf_x_s = lfx_scl * n_x_d + bias_x;
                    double lf_y_s = lfy_scl * n_y_d + bias_y;

                    auto n_x_s = int(std::round(lf_x_s));
                    auto n_y_s = int(std::round(lf_y_s));

                    auto inner = n_x_s >= 0 && n_x_s < src_width - 1 &&
                            n_y_s >= 0 && n_y_s < src_height - 1;

                    if (!inner) {
                        for (int c = 0; c < channels; c++) {
                            dst_im[(n_y_d * dst_width + n_x_d) * channels + c] = outer_value;
                        }
                        continue;
                    }

                    for (int c = 0; c < channels; c++) {
                        dst_im[(n_y_d * dst_width + n_x_d) * channels + c] = src_im[(n_y_s * src_width + n_x_s) * channels + c];
                    }//end for c
                }
            }

        }

        template<typename T>
        static void affine_sample2d_cubic(const Tensor *x, Tensor *y, int x_height, int x_width,
                                 int y_height, int y_width,
                                 unsigned int x_offset, unsigned int y_offset,
                                 int channels, float rz00, float rz01, float rz02, float rz10,
                                 float rz11, float rz12, float rz20, float rz21, float rz22,
                                 T outer_value = T(0)) {
            const T *psrc = x->data<T>() + x_offset;
            T *pdst = y->data<T>() + y_offset;
            const double A = -0.75f;
            double coeffsY[4];
            double coeffsX[4];
            int srcrows = x_width * channels;
            int dstrows = y_width * channels;

            for (int m = 0; m < y_height; m++) {
                for (int n = 0; n < y_width; n++) {
                    vec3d<T> cur;
                    cur.x = n;
                    cur.y = m;
                    cur.z = 1;
                    vec3d<T> location = transform<T>(rz00, rz01, rz02, rz10, rz11, rz12, rz20, rz21, rz22, cur);


                    double fy = location.y;
                    int sy = std::floor(fy);
                    fy -= sy;

                    double fx = location.x;
                    int sx = std::floor(fx);
                    fx -= sx;

                    auto outter = sy < 1 || sy >= x_height - 3 || sx < 1 || sx >= x_width - 3;

                    if (outter) {
                        for (int k = 0; k < channels; k++) {
                            pdst[m * dstrows + n * channels + k] = outer_value;
                        }
                        continue;
                    }

                    coeffsY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
                    coeffsY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
                    coeffsY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
                    coeffsY[3] = 1.f - coeffsY[0] - coeffsY[1] - coeffsY[2];

                    coeffsX[0] = ((A * (fx + 1) - 5 * A) * (fx + 1) + 8 * A) * (fx + 1) - 4 * A;
                    coeffsX[1] = ((A + 2) * fx - (A + 3)) * fx * fx + 1;
                    coeffsX[2] = ((A + 2) * (1 - fx) - (A + 3)) * (1 - fx) * (1 - fx) + 1;
                    coeffsX[3] = 1.f - coeffsX[0] - coeffsX[1] - coeffsX[2];

                    for (int k = 0; k < channels; k++) {
                        pdst[m * dstrows + n * channels + k] = (T) ((
                                psrc[(sy - 1) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[0] +
                                psrc[(sy) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[1] +
                                psrc[(sy + 1) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[2] +
                                psrc[(sy + 2) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[3] +

                                psrc[(sy - 1) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[0] +
                                psrc[(sy) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[1] +
                                psrc[(sy + 1) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[2] +
                                psrc[(sy + 2) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[3] +

                                psrc[(sy - 1) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[0] +
                                psrc[(sy) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[1] +
                                psrc[(sy + 1) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[2] +
                                psrc[(sy + 2) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[3] +

                                psrc[(sy - 1) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[0] +
                                psrc[(sy) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[1] +
                                psrc[(sy + 1) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[2] +
                                psrc[(sy + 2) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[3]));

                    }

                }

            }
        }


        template<typename T>
        static void batch_affine_sample2d(int number, const Tensor *x, Tensor *y, int x_height, int x_width,
                                 int y_height, int y_width,
                                 unsigned int x_batch_step, unsigned int y_batch_step,
                                 int channels,Affine_Sample2DType type, float rz00, float rz01, float rz02, float rz10,
                                 float rz11, float rz12, float rz20, float rz21, float rz22,
                                 T outer_value = T(0)) {
             if(type == Affine_Sample2DType::CUBIC) {
                 for(int k=0; k<number; k++) {
                     affine_sample2d_cubic<T>(x,y,x_height,x_width,y_height,y_width,k*x_batch_step, k*y_batch_step, channels,
                                               rz00,rz01,rz02,rz10,rz11,rz12,rz20,rz21,rz22, outer_value);
                 } 
             }else if(type == Affine_Sample2DType::NEAREST) {

                 for(int k=0; k<number; k++) {
                     affine_sample2d_nearest<T>(x,y,x_height,x_width,y_height,y_width,k*x_batch_step, k*y_batch_step, channels,
                                               rz00,rz01,rz02,rz10,rz11,rz12,rz20,rz21,rz22, outer_value);
                 } 
             }else { //LINEAR
                 for(int k=0; k<number; k++) {
                     affine_sample2d_linear<T>(x,y,x_height,x_width,y_height,y_width,k*x_batch_step, k*y_batch_step, channels,
                                               rz00,rz01,rz02,rz10,rz11,rz12,rz20,rz21,rz22, outer_value);
                 } 
             }
        }

        void Affine_Sample2D::affine_sample_run(const Tensor &x, float rz00, float rz01, float rz02, float rz10,
                                           float rz11, float rz12, float rz20, float rz21, float rz22,
                                           Affine_Sample2DType type, int dim, float outer_value,
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
                        x_batch_step, y_batch_step, channels, type, rz00,rz01,rz02,rz10,rz11,rz12,rz20,rz21,rz22, outer_value); break; }
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
                    TS_LOG_ERROR << this->op() << " not support this data type: " << type_str(dtype) << eject;
                    break;
                }
            }

        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Affine_Sample2D, CPU, name::layer::affine_sample2d())
