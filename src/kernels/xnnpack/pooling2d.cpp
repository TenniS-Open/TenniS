#include <runtime/runtime.h>
#include "backend/common_structure.h"
#include "kernels/xnnpack/conv2d.h"
#include "kernels/xnnpack/xnnpack.h"
#include "pooling2d.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "backend/common_function.h"
#include "global/operator_factory.h"
#include "utils/assert.h"
#include "utils/log.h"

namespace ts {
    namespace xnn {
        Pooling2D::Pooling2D() {
            field(name::format, REQUIRED);
            field(name::type, REQUIRED);
            field(name::padding, REQUIRED);
            field(name::padding_type, OPTIONAL, tensor::from(int(Padding2DType::BLACK)));
            field(name::ksize, REQUIRED);
            field(name::stride, REQUIRED);
        }

        static std::string to_string(const std::valarray<int> &arr) {
            std::ostringstream out;
            out << "[";
            for (size_t i = 0; i < arr.size(); ++i) {
                if (i) out << ", ";
                out << arr[i];
            }
            out << "]";
            return out.str();
        }

        void Pooling2D::init() {
            supper::init();

            auto format = tensor::to_string(get(name::format));
            m_type = static_cast<Pooling2DType>(tensor::to_int(get(name::type)));
            auto padding_tensor = tensor::cast(INT32, get(name::padding));
            m_padding_type = static_cast<Padding2DType>(tensor::to_int(get(name::padding_type)));
            auto ksize_tensor = tensor::cast(INT32, get(name::ksize));
            auto stride_tensor = tensor::cast(INT32, get(name::stride));

            TS_AUTO_CHECK(padding_tensor.has_shape({4, 2}));
            TS_AUTO_CHECK(ksize_tensor.has_shape({4,}));
            TS_AUTO_CHECK(stride_tensor.has_shape({4,}));

            if (format == name::NCHW) {
                m_format = FORMAT_NCHW;
            }
            else if (format == name::NHWC) {
                m_format = FORMAT_NHWC;
            } else {
                TS_LOG_ERROR << this->op() << " do not support format: " << format << eject;
            }

            m_padding4x2.resize(8);
            for (size_t i = 0; i < 8; ++i) m_padding4x2[i] = padding_tensor.data<int32_t>(i);
            m_ksize4.resize(4);
            for (size_t i = 0; i < 4; ++i) m_ksize4[i] = ksize_tensor.data<int32_t>(i);
            m_stride4.resize(4);
            for (size_t i = 0; i < 4; ++i) m_stride4[i] = stride_tensor.data<int32_t>(i);

            // only support native conv2d
            if (m_format == FORMAT_NCHW) {
                if (m_padding4x2[0] != 0 ||
                    m_padding4x2[1] != 0 ||
                    m_padding4x2[2] != 0 ||
                    m_padding4x2[3] != 0) {
                    TS_LOG_ERROR << this->op() << " do not support padding: " << to_string(m_padding4x2) << eject;
                }
                if (m_ksize4[0] != 1 ||
                    m_ksize4[1] != 1) {
                    TS_LOG_ERROR << this->op() << " do not support ksize: " << to_string(m_ksize4) << eject;
                }
                if (m_stride4[0] != 1 ||
                    m_stride4[1] != 1) {
                    TS_LOG_ERROR << this->op() << " do not support stride: " << to_string(m_stride4) << eject;
                }
            } else if (m_format == FORMAT_NHWC) {
                if (m_padding4x2[0] != 0 ||
                    m_padding4x2[1] != 0 ||
                    m_padding4x2[6] != 0 ||
                    m_padding4x2[7] != 0) {
                    TS_LOG_ERROR << this->op() << " do not support padding: " << to_string(m_padding4x2) << eject;
                }
                if (m_ksize4[0] != 1 ||
                    m_ksize4[3] != 1) {
                    TS_LOG_ERROR << this->op() << " do not support ksize: " << to_string(m_ksize4) << eject;
                }
                if (m_stride4[0] != 1 ||
                    m_stride4[3] != 1) {
                    TS_LOG_ERROR << this->op() << " do not support stride: " << to_string(m_stride4) << eject;
                }
            }
        }

        int Pooling2D::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto x_tensor = stack[0];

            TS_AUTO_CHECK(x_tensor.dims() == 4);

            Size2D x;
            Size2D ksize;
            Padding2D padding;
            Stride2D stride;

            x = Size2D(x_tensor.size(1), x_tensor.size(2));
            ksize = Size2D(m_ksize4[2], m_ksize4[3]);
            padding = Padding2D(m_padding4x2[4], m_padding4x2[5], m_padding4x2[6], m_padding4x2[7]);
            stride = Stride2D(m_stride4[2], m_stride4[3]);

            Size2D y = pooling2d_forward(x, padding, ksize, stride);

            Tensor::Prototype out_proto;

            out_proto = Tensor::Prototype(x_tensor.dtype(), {x_tensor.size(0), y.height, y.width, x_tensor.size(3)});

            output.resize(1);
            output[0] = out_proto;

            return 1;
        }

        int Pooling2D::run(Stack &stack) {
            std::vector<Tensor::Prototype> output_protos;
            infer(stack, output_protos);

            auto memory_device = running_memory_device();

            Tensor x = stack[0].view(memory_device);

            Tensor out = *stack.push(output_protos[0], memory_device);

            Size2D ksize = Size2D(m_ksize4[2], m_ksize4[3]);
            Padding2D padding = Padding2D(m_padding4x2[4], m_padding4x2[5], m_padding4x2[6], m_padding4x2[7]);
            Stride2D stride = Stride2D(m_stride4[2], m_stride4[3]);
            Padding2DType padding_type = m_padding_type;

            pooling2d(x, m_type, padding, padding_type, ksize, stride, m_format, out);

            return 1;
        }

        void Pooling2D::pooling2d(const Tensor &x, Pooling2DType type,
                                   const Padding2D &padding, Padding2DType padding_type,
                                   const Size2D &ksize, const Stride2D &stride,
                                   Conv2DFormat format, Tensor &out) {
            if (padding_type != Padding2DType::BLACK) {
                TS_LOG_INFO << "padding type got " << static_cast<int>(padding_type) << ", but using black";
            }
            size_t channels = x.size(3);
            size_t input_stride = channels;
            size_t output_stride = channels;
            float min = -std::numeric_limits<float>::infinity();
            float max = std::numeric_limits<float>::infinity();
            if (m_type == Pooling2DType::AVG) {
                TS_LOG_INFO << "enter average pooling2d at " << this->name();
                if (m_op == nullptr) {
                    auto ctx = ctx::get<RuntimeContext>();
                    m_threadpool = ctx->get_xnn_threadpool();

                    m_status = xnn_create_average_pooling2d_nhwc_f32(padding.top, padding.right, padding.bottom, padding.left, ksize.height, ksize.width, stride.height, stride.width,
                                    channels, input_stride, output_stride, min, max, 0, &m_op);
                    TS_CHECK_EQ(m_status, xnn_status_success);
                    m_shared_op.reset(m_op, xnn_delete_operator);
                    m_op = m_shared_op.get();
                }
                m_status = xnn_setup_average_pooling2d_nhwc_f32(m_op, x.size(0), x.size(1), x.size(2), x.data<float>(), out.data<float>(), m_threadpool);
                TS_CHECK_EQ(m_status, xnn_status_success);
                m_status = xnn_run_operator(m_op, m_threadpool);
                TS_CHECK_EQ(m_status, xnn_status_success);
            } else if (m_type == Pooling2DType::MAX) {
                if (m_op == nullptr) {
                    auto ctx = ctx::get<RuntimeContext>();
                    m_threadpool = ctx->get_xnn_threadpool();

                    m_status = xnn_create_max_pooling2d_nhwc_f32(padding.top, padding.right, padding.bottom, padding.left, ksize.height, ksize.width, stride.height, stride.width,
                                    1, 1, channels, input_stride, output_stride, min, max, 0, &m_op);
                    TS_CHECK_EQ(m_status, xnn_status_success);
                    m_shared_op.reset(m_op, xnn_delete_operator);
                    m_op = m_shared_op.get();
                }
                m_status = xnn_setup_max_pooling2d_nhwc_f32(m_op, x.size(0), x.size(1), x.size(2), x.data<float>(), out.data<float>(), m_threadpool);
                TS_CHECK_EQ(m_status, xnn_status_success);
                m_status = xnn_run_operator(m_op, m_threadpool);
                TS_CHECK_EQ(m_status, xnn_status_success);
            } else {
                TS_LOG_ERROR << "Only support average pooling2d and max pooling2d" << eject;
            }
        }
    }
}
using namespace ts;
using namespace xnn;
TS_REGISTER_OPERATOR(Pooling2D, ts::XNNPACK, "xnn::pooling2d")
