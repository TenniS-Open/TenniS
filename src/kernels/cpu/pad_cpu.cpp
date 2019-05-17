#include <kernels/cpu/pad_cpu.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

#include <core/memory.h>
#include <numeric>

#ifdef TS_USE_OPENMP
#include "kernels/common/openmp.h"
#endif

namespace ts {
    namespace cpu {
        /**
         *
         * @param [in] out_coord the coordinate in output map
         * @param [in] padding padding
         * @param [in] x_shape the input shape
         * @param [out] x_coord the transpose back coordinate
         * @return true if transposed, else return false
         */
        static inline bool transpose_pad(const std::vector<int> &out_coord,
                const std::vector<std::array<int, 2>> &padding,
                const std::vector<int> &x_shape,
                std::vector<int> &x_coord) {
            auto dims = out_coord.size();
            x_coord.resize(dims);
            for (size_t i = 0; i < dims; ++i) {
                auto offset = out_coord[i] - padding[i][0];
                if (offset < 0 || offset >= x_shape[i]) return false;
                x_coord[i] = offset;
            }
            return true;
        }

        template <typename T>
        static inline void cut2d(const Tensor &x, const std::array<int, 2> &padding_h, const std::array<int, 2> &padding_w,
            float padding_value, Tensor &out) {
            int cut_top = padding_h[0];
            int cut_bottom = padding_h[1];
            int cut_left = padding_w[0];
            int cut_right = padding_w[1];

            auto src_shape = x.sizes();
            int num = src_shape[0];
            int channel = src_shape[1];
            int src_h = src_shape[2];
            int src_w = src_shape[3];
            int src_channel_offset = src_h * src_w;
            int src_num_offset = src_shape[1] * src_channel_offset;

            auto dst_shape = out.sizes();
            dst_shape[0] = num;
            dst_shape[1] = channel;
            int out_h = src_h + cut_top + cut_bottom;
            int out_w = src_w + cut_left + cut_right;
            dst_shape[2] = out_h;
            dst_shape[3] = out_w;
            out.reshape(dst_shape);
            int dst_channel_offset = out_h * out_w;
            int dst_num_offset = channel * dst_channel_offset;

            const T* src_data = x.data<T>();
            T* dst_data = out.data<T>();

            for (int n = 0; n < num; n++) {
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < channel; c++) {
                    const T* src_at = src_data + n * src_num_offset + c * src_channel_offset;
                    T* dst_at = dst_data + n * dst_num_offset + c * dst_channel_offset;

                    const T* cut_at = src_at - cut_top * src_w - cut_left;

                    for (int h = 0; h < out_h; h++) {
                        //for (int w = 0; w < out_w; w++)
                        //{
                        //    dst_at[w] = cut_at[w];
                        //}
                        //NOTE:ts::memcpy is very slow
                        if (out_w <12) {
                            for (int w = 0; w < out_w; w++)
                            {
                                dst_at[w] = cut_at[w];
                            }
                        }
                        else {
                            //memcpy(dst_at, out.device(), size_t(out_w * out.proto().type_bytes()),
                                //cut_at, x.device(), size_t(out_w * x.proto().type_bytes()));
                            std::memcpy(dst_at, cut_at, out_w * sizeof(T));
                        }

                        dst_at += out_w;
                        cut_at += src_w;
                    }
                }
            }
        }

        template <typename T>
        static inline void pad2d(const Tensor &x, const std::array<int, 2> &padding_h, const std::array<int, 2> &padding_w, 
            float padding_value, Tensor &out) {
            int pad_top = padding_h[0];
            int pad_bottom = padding_h[1];
            int pad_left = padding_w[0];
            int pad_right = padding_w[1];

            auto src_shape = x.sizes();
            int num = src_shape[0];
            int channel = src_shape[1];
            int src_h = src_shape[2];
            int src_w = src_shape[3];
            int src_channel_offset = src_h * src_w;
            int src_num_offset = src_shape[1] * src_channel_offset;

            auto dst_shape = out.sizes();
            dst_shape[0] = num;
            dst_shape[1] = channel;
            int out_h = src_h + pad_top + pad_bottom;
            int out_w = src_w + pad_left + pad_right;
            dst_shape[2] = out_h;
            dst_shape[3] = out_w;
            out.reshape(dst_shape);
            int dst_channel_offset = out_h * out_w;
            int dst_num_offset = channel * dst_channel_offset;

            const T* src_data = x.data<T>();
            T* dst_data = out.data<T>();
            T value = padding_value;
            auto out_device = out.device();
            auto x_device = x.device();

            for (int n = 0; n < num; n++){
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < channel; c++){
                    const T* src_at = src_data + n * src_num_offset + c * src_channel_offset;
                    T* dst_at = dst_data + n * dst_num_offset + c * dst_channel_offset;

                    //fill top
                    int h = 0;
                    for (; h < pad_top; h++){
                        int w = 0;
                        for (; w < out_w; w++){
                            dst_at[w] = value;
                        }
                        dst_at += out_w;
                    }

                    //fill center
                    for (; h < pad_top + src_shape[2]; h++){
                        int w = 0;
                        for (; w < pad_left; w++){
                            dst_at[w] = value;
                        }
                        //for (; w < (pad_left + src_w); w++) {
                        //    dst_at[w] = src_at[w - pad_left];
                        //}
                        //NOTE:12 is an empirical data,but ts::memcpy is very slow,sad
                        if (src_w < 12){
                            for (; w < (pad_left + src_w); w++){
                                dst_at[w] = src_at[w - pad_left];
                            }
                        }
                        else{
                            //memcpy(dst_at + pad_left, out_device, size_t(src_w * out.proto().type_bytes()),
                            //    src_at, x_device, size_t(src_w * x.proto().type_bytes()));
                            std::memcpy(dst_at + pad_left, src_at, src_w * sizeof(T));
                            w += src_w;
                        }
                        for (; w < out_w; w++){
                            dst_at[w] = value;
                        }
                        dst_at += out_w;
                        src_at += src_w;
                    }
                    // fill bottom
                    for (; h < out_h; h++){
                        int w = 0;
                        for (; w < out_w; w++){
                            dst_at[w] = value;
                        }
                        dst_at += out_w;
                    }
                }
            }
        }

        template <typename T>
        static inline void cpu_pad_compute_run(const Tensor &x, const std::vector<std::array<int, 2>> &padding, float padding_value, Tensor &out) {
            int left = 0;
            while (left < padding.size()) {
                if (padding[left][0] != 0 || padding[left][1] != 0) break;
                ++left;
            }
            if (left >= padding.size()) {
                memcpy(out.data(), out.device(), size_t(out.count() * out.proto().type_bytes()),
                    x.data(), x.device(), size_t(x.count() * x.proto().type_bytes()));
                return;
            }

            int right = int(padding.size()) - 1;
            while (right > left) {
                if (padding[right][0] != 0 || padding[right][1] != 0) break;
                --right;
            }

            //NOTE:pad2d and cut2d only support NCHW pad on [H,W] now.
            //pad2d don't support negative pad,cut2d support negative pad only.
            if (padding.size() == 4 && padding[2][0] > 0 && padding[2][1] > 0 && padding[3][0] > 0 && padding[3][1] > 0) {
                if (left == 2 && right == 3) {
                    pad2d<T>(x, padding[left], padding[right], padding_value, out);
                    return;
                }
            }
            
            if (padding.size() == 4 && padding[2][0] <= 0 && padding[2][1] <= 0 && padding[3][0] <= 0 && padding[3][1] <= 0) {
                if (left == 2 && right == 3) {
                    cut2d<T>(x, padding[left], padding[right], padding_value, out);
                    return;
                }
            }

            ShapeIterator out_it(out.sizes());
            auto &x_shape = x.sizes();
            std::vector<int> x_coord;
            auto out_data = out.data<T>();
            auto x_data = x.data<T>();

            HypeShape hype_x_shape(x_shape);

            auto value = T(padding_value);

            int count = out.count();
            for (int i = 0; i < count; ++i) {
                bool not_overflow = transpose_pad(out_it.coordinate(), padding, x_shape, x_coord);
                if (not_overflow) {
                    out_data[i] = x_data[hype_x_shape.to_index(x_coord)];
                } else {
                    out_data[i] = value;
                }
                ++out_it;
            }
        }

        void PadOnCPU::pad(const Tensor &x, const std::vector<std::array<int, 2>> &padding, float padding_value, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch(dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_pad_compute_run<TYPE>(x, padding, padding_value, out); break; }
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
                    TS_LOG_ERROR << "pad not support this data type: " << dtype << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(PadOnCPU, CPU, name::layer::pad())
