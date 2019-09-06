//
// Created by kier on 2019-09-03.
//

#include "runtime/operator.h"
#include "global/operator_factory.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "runtime/stack.h"
#include "kernels/cpu/operator_on_cpu.h"
#include "backend/base/base_roi_align.h"

/**
 * THis cpu implement was fork from Dragon, SeetaTech
 */

namespace ts {
    namespace cpu {
        template<typename T>
        void ROIAlign_kernel(
                const int C,
                const int H,
                const int W,
                const int pool_h,
                const int pool_w,
                const int num_rois,
                const float spatial_scale,
                const int sampling_ratio,
                const T *x,
                const T *rois,
                T *y) {
            TS_LOG_ERROR << "Not implement." << eject;
        }

        template<typename T>
        T _ROIAlignInterpolate(
                const T *Xdata,
                const int height,
                const int width,
                T y,
                T x) {
            if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;
            if (y <= 0) y = 0;
            if (x <= 0) x = 0;

            int y_low = (int) y;
            int x_low = (int) x;
            int y_high;
            int x_high;

            if (y_low >= height - 1) {
                y_high = y_low = height - 1;
                y = (T) y_low;
            } else {
                y_high = y_low + 1;
            }

            if (x_low >= width - 1) {
                x_high = x_low = width - 1;
                x = (T) x_low;
            } else {
                x_high = x_low + 1;
            }

            T ly = y - y_low;
            T lx = x - x_low;
            T hy = (T) 1 - ly, hx = (T) 1 - lx;
            T v1 = Xdata[y_low * width + x_low];
            T v2 = Xdata[y_low * width + x_high];
            T v3 = Xdata[y_high * width + x_low];
            T v4 = Xdata[y_high * width + x_high];
            T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
            T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
            return val;
        }

        template<>
        void ROIAlign_kernel<float>(
                const int C,
                const int H,
                const int W,
                const int pool_h,
                const int pool_w,
                const int num_rois,
                const float spatial_scale,
                const int sampling_ratio,
                const float *x,
                const float *rois,
                float *y) {
            const int64_t X_offset = H * W, Y_offset = pool_h * pool_w;
            const int64_t x_offset = C * X_offset, y_offset = C * Y_offset;

            for (int n = 0; n < num_rois; ++n) {
                auto *R = rois + n * 5;
                int roi_batch_ind = (int) R[0];
                auto *Y = y + n * y_offset;

                if (roi_batch_ind < 0) {
                    std::memset(Y, 0, sizeof(float) * y_offset);
                    continue;
                }

                float roi_start_w = R[1] * spatial_scale;
                float roi_start_h = R[2] * spatial_scale;
                float roi_end_w = R[3] * spatial_scale;
                float roi_end_h = R[4] * spatial_scale;

                float roi_width = std::max(roi_end_w - roi_start_w, 1.f);
                float roi_height = std::max(roi_end_h - roi_start_h, 1.f);
                float bin_size_h = (float) roi_height / (float) pool_h;
                float bin_size_w = (float) roi_width / (float) pool_w;

                int roi_bin_grid_h = (sampling_ratio > 0) ?
                                     sampling_ratio : (int) ceil(roi_height / pool_h);
                int roi_bin_grid_w = (sampling_ratio > 0) ?
                                     sampling_ratio : (int) ceil(roi_width / pool_w);

                const float num_bin_grids = (float) roi_bin_grid_h * roi_bin_grid_w;
                const float *X = x + roi_batch_ind * x_offset;

                for (int c = 0; c < C; ++c) {
                    for (int ph = 0; ph < pool_h; ++ph) {
                        for (int pw = 0; pw < pool_w; ++pw) {
                            float output_val = 0.f;
                            const int pool_idx = ph * pool_w + pw;
                            for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                                const float y = roi_start_h + ph * bin_size_h +
                                                static_cast<float>(iy + .5f) * bin_size_h /
                                                static_cast<float>(roi_bin_grid_h);
                                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                                    const float x = roi_start_w + pw * bin_size_w +
                                                    static_cast<float>(ix + .5f) * bin_size_w /
                                                    static_cast<float>(roi_bin_grid_w);
                                    output_val += _ROIAlignInterpolate<float>(X, H, W, y, x);
                                }  // End ix
                            }  // End iy
                            output_val /= num_bin_grids;
                            Y[pool_idx] = output_val;
                        }  // End pw
                    }  // End ph
                    // Offset according to C
                    X += X_offset;
                    Y += Y_offset;
                }  // End c
            }  // End n
        }

        template <typename T>
        void ROIAlign_run_with_type(
                std::vector<Tensor> &outputs,
                const std::vector<Tensor> &inputs,
                int pool_h, int pool_w, float spatial_scale, int sampling_ratio) {
            auto* Xdata = inputs[0].data<T>();
            auto* Rdata = inputs[1].data<T>();
            auto* Ydata = outputs[0].data<T>();

            ROIAlign_kernel(
                    inputs[0].size(1), inputs[0].size(2), inputs[0].size(3),
                    pool_h, pool_w, inputs[1].size(0),
                    spatial_scale, sampling_ratio,
                    Xdata, Rdata, Ydata);
        }

        class ROIAlign : public OperatorOnCPU<base::ROIAlign> {
        public:
            std::vector<Tensor> roi_align(
                    const std::vector<Tensor> &inputs,
                    int pool_h, int pool_w, float spatial_scale, int sampling_ratio) override {
                DTYPE dtype = inputs[0].dtype();
                std::vector<Tensor> outputs(1);
                outputs[0] = Tensor(Tensor::InFlow::HOST, dtype,
                        {inputs[1].size(0), inputs[0].size(1), pool_h, pool_w});
                switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { ROIAlign_run_with_type<TYPE>(outputs, inputs, pool_h, pool_w, spatial_scale, sampling_ratio); break; }
                    DECLARE_COMPUTE_RUN(FLOAT32, float);
                    // DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                    default: {
                        TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                        break;
                    }
                }
                return outputs;
            }
        };
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(ROIAlign, CPU, name::layer::roi_align())
