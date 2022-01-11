//
// Created by sen on 2022/1/6.
//
#include "transpose.h"
#include "kernels/common/simd.h"
#include "utils/ctxmgr_lite.h"
#include "kernels/xnnpack/threadpool.h"
#include "backend/name.h"
#include "core/device.h"
#include "global/operator_factory.h"
#include "runtime/runtime.h"

namespace ts {
    namespace xnn {
        void Transpose::init() {
            supper::init();
            auto ctx = ctx::get<RuntimeContext>();
            m_pool = ctx->get_xnn_threadpool();
        }

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

                pdst[index] = psrc[i];
                ++input_shape_it;
            }
        }

        template<typename T>
        struct NC3HWToNHWC3_kernel_context {
            const T *src_at;
            T *dst_at;
            int width;
            int channel;
            int channel_offset;
        };

        template<typename T>
        void NC3HWToNHWC3_kernel(struct NC3HWToNHWC3_kernel_context<T> *context, size_t h) {
            auto src_tmp = context->src_at + h * context->width;
            auto dst_tmp = context->dst_at + h * context->channel * context->width;
            for (int w = 0; w < context->width; ++w) {
                int dst_base_offset = w * context->channel;
                for (int c = 0; c < context->channel; ++c) {
                    dst_tmp[dst_base_offset + c] = src_tmp[c * context->channel_offset + w];
                }
            }
        }

        template<typename T>
        void NC3HWToNHWC3_simd_kernel(struct NC3HWToNHWC3_kernel_context<T> *context, size_t h) {
            auto src_tmp = context->src_at + h * context->width;
            auto dst_tmp = context->dst_at + h * context->channel * context->width;
            int w = 0;
            for (; w + 3 < context->width; w += 4) {
                float32x4 i0(src_tmp), i1(src_tmp + context->channel_offset), i2(src_tmp + 2 * context->channel_offset);
                float32x4x3 ix4x3(i0, i1, i2);
                incx4x3_save(dst_tmp, ix4x3);

                src_tmp += 4;
                dst_tmp += 12;
            }
            for (; w < context->width; ++w) {
                int dst_at_base_offset = h * context->width * context->channel + w * context->channel;
                int src_at_base_offset = h * context->width + w;
                for (int c = 0; c < context->channel; ++c) {
                    context->dst_at[dst_at_base_offset + c] = context->src_at[src_at_base_offset +
                                                                              c * context->channel_offset];
                }
            }
        }

        template<typename T>
        static inline void NC3HWToNHWC3(const T *psrc, T *pdst, const Shape &input_shape, pthreadpool_t pool) {
            int height = input_shape[2], width = input_shape[3], channel = input_shape[1];
            int channel_offset = height * width;
            int num_offset = channel * channel_offset;
            for (int n = 0; n < input_shape[0]; ++n) {
                const T *src_at = psrc + n * num_offset;
                T *dst_at = pdst + n * num_offset;

                NC3HWToNHWC3_kernel_context<T> kernel_context{
                        src_at, dst_at, width, channel, channel_offset
                };
                auto pool_ctx = ctx::get<XnnThreadPool>();
                pthreadpool_parallelize_1d(pool,
                                           (pthreadpool_task_1d_t) NC3HWToNHWC3_kernel<T>,
                                           (void **) &kernel_context, height, 0);
            }
        }

        template<>
        inline void NC3HWToNHWC3<float>(const float *psrc, float *pdst, const Shape &input_shape, pthreadpool_t pool) {
            int height = input_shape[2], width = input_shape[3], channel = input_shape[1];
            int channel_offset = height * width;
            int num_offset = channel * channel_offset;
            for (int n = 0; n < input_shape[0]; ++n) {
                const float *src_at = psrc + n * num_offset;
                float *dst_at = pdst + n * num_offset;

                NC3HWToNHWC3_kernel_context<float> kernel_context{
                        src_at, dst_at, width, channel, channel_offset
                };
                pthreadpool_parallelize_1d(pool,
                                           (pthreadpool_task_1d_t) NC3HWToNHWC3_simd_kernel<float>,
                                           (void **) &kernel_context, height, 0);
            }
        }

        template<typename T>
        struct NHWC3ToNC3HW_kernel_context {
                    const T *src_at;
                    T *dst_at;
                    int h_offset;
                    int width;
                    int channel;
                    int channel_offset;
                };

        template<typename T>
        static inline void NHWC3ToNC3HW_kernel(struct NHWC3ToNC3HW_kernel_context<T> *context, size_t h) {
            auto src_tmp = context->src_at + h * context->h_offset;
            auto dst_tmp = context->dst_at + h * context->width;
            for (int w = 0; w < context->width; ++w) {
                int src_tmp_base_offset = w * context->channel;
                for (int c = 0; c < context->channel; ++c) {
                    dst_tmp[c * context->channel_offset + w] = src_tmp[src_tmp_base_offset + c];
                }
            }
        }

        template<typename T>
        static inline void NHWC3ToNC3HW(const T *psrc, T *pdst, const Shape &input_shape, pthreadpool_t pool) {
            int height = input_shape[1], width = input_shape[2], channel = input_shape[3];
            int h_offset = width * channel;
            int channel_offset = height * width;
            int num_offset = height * h_offset;
            for (int n = 0; n < input_shape[0]; ++n) {
                const T *src_at = psrc + n * num_offset;
                T *dst_at = pdst + n * num_offset;

                NHWC3ToNC3HW_kernel_context<T> kernel_context {
                    src_at, dst_at, h_offset, width, channel, channel_offset
                };
                auto pool_ctx = ctx::get<XnnThreadPool>();
                pthreadpool_parallelize_1d(pool,
                                           (pthreadpool_task_1d_t) NHWC3ToNC3HW_kernel<T>,
                                           (void**)&kernel_context, height, 0);
            }
        }

        template<typename T>
        static inline void NHWC3ToNC3HW_simd_kernel(struct NHWC3ToNC3HW_kernel_context<T> *context, size_t h) {
            auto src_tmp = context->src_at + h * context->h_offset;
            auto dst_tmp = context->dst_at + h * context->width;

            int w = 0;
            for (; w + 3 < context->width; w += 4) {
                float32x4x3 in_x4x3 = incx4x3_load(src_tmp, 3);
                in_x4x3.store(dst_tmp, 0);
                in_x4x3.store(dst_tmp + context->channel_offset, 1);
                in_x4x3.store(dst_tmp + 2 * context->channel_offset, 2);

                src_tmp += 12;
                dst_tmp += 4;
            }
            for (; w < context->width; ++w) {
                int dst_at_base_offset = h * context->width + w;
                int src_at_base_offset = h * context->h_offset + w * context->channel;
                for (int c = 0; c < context->channel; ++c) {
                    context->dst_at[dst_at_base_offset + c * context->channel_offset] = context->src_at[src_at_base_offset + c];
                }
            }

        }

        template<>
        inline void NHWC3ToNC3HW<float>(const float *psrc, float *pdst, const Shape &input_shape, pthreadpool_t pool) {
            int height = input_shape[1], width = input_shape[2], channel = input_shape[3];
            int h_offset = width * channel;
            int channel_offset = height * width;
            int num_offset = height * h_offset;
            for (int n = 0; n < input_shape[0]; ++n) {
                const float *src_at = psrc + n * num_offset;
                float *dst_at = pdst + n * num_offset;

                NHWC3ToNC3HW_kernel_context<float> kernel_context {
                    src_at, dst_at, h_offset, width, channel, channel_offset
                };
                auto pool_ctx = ctx::get<XnnThreadPool>();
                pthreadpool_parallelize_1d(pool,
                                           (pthreadpool_task_1d_t)NHWC3ToNC3HW_simd_kernel<float>,
                                           (void**)&kernel_context, height, 0);
            }
        }

        template<typename T>
        static inline void cpu_transpose_compute_run(const Tensor &x, const std::vector<int> &permute, Tensor &out, pthreadpool_t pool) {
            //NOTE: optimize NHWC(C==3)->NCHW.
            //TODO: optimize NHWC(C==4)->NCHW.
            auto input_shape = x.sizes();
            std::vector<int> NHWC3ToNC3HW_order{0, 3, 1, 2};
            std::vector<int> NCHW3ToNHWC3_order{0, 2, 3, 1};
            if (permute == NHWC3ToNC3HW_order && input_shape[3] == 3)
                NHWC3ToNC3HW(x.data<T>(), out.data<T>(), x.sizes(), pool);
            else if (permute == NCHW3ToNHWC3_order && input_shape[1] == 3) {
                NC3HWToNHWC3(x.data<T>(), out.data<T>(), x.sizes(), pool);
            } else
                Transpose_transpose_run(x.data<T>(), out.data<T>(), x.count(), permute, x.sizes(), out.sizes());
        }

        void Transpose::transpose(const Tensor &x, const std::vector<int> &permute, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_transpose_compute_run<TYPE>(x, permute, out, m_pool); break; }
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
TS_REGISTER_OPERATOR(Transpose, ts::XNNPACK, name::layer::transpose())
