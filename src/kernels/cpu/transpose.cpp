#include <kernels/cpu/transpose.h>
#include <set>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>
#include <core/tensor_builder.h>
#include <kernels/common/simd.h>
#ifdef TS_USE_OPENMP
#include <kernels/common/openmp.h>
#endif

namespace ts {
    namespace cpu {

        template<typename T>
        static void Transpose_transpose_run(
                const T *psrc, T *pdst, int len,
                const std::vector<int> &permute,
                const Shape &input_shape, const Shape &output_shape) {
            Shape tmpshape(input_shape.size());

            // HypeShape hype_input_shape(input_shape);
            HypeShape hype_output_shape(output_shape);
            ShapeIterator input_shape_it(input_shape);

            int index = 0;
            for (int i = 0; i < len; i++) {
                auto &oldshape = input_shape_it.coordinate();
                for (size_t k = 0; k < oldshape.size(); k++) {
                    tmpshape[k] = oldshape[permute[k]];
                }

                index = hype_output_shape.to_index(tmpshape);//to_index(reshape, tmpshape);

                //if(index < 0) {
                //    throw ts::Exception("transpose operator failed, index is invalid");
                //}
                pdst[index] = psrc[i];
                ++input_shape_it;
            }
        }

        template<typename T>
        static inline void NHWC3ToNC3HW(const T *psrc, T *pdst, const Shape &input_shape){
            int height = input_shape[1],width = input_shape[2],channel = input_shape[3];
            int h_offset = width * channel;
            int channel_offset = height * width;
            int num_offset = height * h_offset;
            for (int n = 0; n < input_shape[0]; ++n) {
                const T* src_at = psrc + n * num_offset;
                T* dst_at = pdst + n * num_offset;
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int h = 0; h < height; ++h) {
                    auto src_tmp = src_at + h * h_offset;
                    auto dst_tmp = dst_at + h * width;
                    for (int w = 0; w < width; ++w) {
                        for (int c = 0; c < channel; ++c) {
                            dst_tmp[c * channel_offset + w] = src_tmp[w * channel + c];
                        }
                    }
                }
            }
        }

        template<>
        inline void NHWC3ToNC3HW<float>(const float *psrc, float *pdst, const Shape &input_shape) {
            int height = input_shape[1],width = input_shape[2],channel = input_shape[3];
            int h_offset = width * channel;
            int channel_offset = height * width;
            int num_offset = height * h_offset;
            for (int n = 0; n < input_shape[0]; ++n) {
                const float *src_at = psrc + n * num_offset;
                float *dst_at = pdst + n * num_offset;
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int h = 0; h < height; ++h) {
                    auto src_tmp = src_at + h * h_offset;
                    auto dst_tmp = dst_at + h * width;

                    int w = 0;
                    for ( ; w + 3 < width; w += 4) {
                        float32x4x3 in_x4x3 = incx4x3_load(src_tmp, 3);
                        in_x4x3.store(dst_tmp, 0);
                        in_x4x3.store(dst_tmp + channel_offset, 1);
                        in_x4x3.store(dst_tmp + 2 * channel_offset, 2);

                        src_tmp += 12;
                        dst_tmp += 4;
                    }
                    for (; w < width; ++w) {
                        for (int c = 0; c < channel; ++c) {
                            dst_at[h * h_offset + c * channel_offset + w] = src_at[h * width + w * channel + c];
                        }
                    }
                }
            }
        }

        template<typename T>
        static inline void cpu_transpose_compute_run(const Tensor &x, const std::vector<int> &permute, Tensor &out) {
            //NOTE: optimize NHWC(C==3)->NCHW.
            //TODO: optimize NHWC(C==4)->NCHW.
            auto input_shape = x.sizes();
            std::vector<int> NHWC3ToNC3HW_order{0, 3, 1, 2};
            if(permute == NHWC3ToNC3HW_order && input_shape[3] == 3)
                NHWC3ToNC3HW(x.data<T>(), out.data<T>(), x.sizes());
            else
                Transpose_transpose_run(x.data<T>(), out.data<T>(), x.count(), permute, x.sizes(), out.sizes());
        }

        void Transpose::transpose(const Tensor &x, const std::vector<int> &permute, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_transpose_compute_run<TYPE>(x, permute, out); break; }
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
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Transpose, CPU, name::layer::transpose())
