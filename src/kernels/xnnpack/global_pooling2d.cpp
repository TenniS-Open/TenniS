#include <runtime/runtime.h>
#include "global_pooling2d.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "backend/common_function.h"
#include "global/operator_factory.h"

namespace ts {
    namespace xnn {
        GlobalPooling2D::GlobalPooling2D() {
            field(name::format, REQUIRED);
            field(name::type, REQUIRED);
        }

        void GlobalPooling2D::init() {
            supper::init();

            auto ctx = ctx::get<RuntimeContext>();
            m_threadpool = ctx->get_xnn_threadpool();

            auto format = tensor::to_string(get(name::format));
            m_type = static_cast<Pooling2DType>(tensor::to_int(get(name::type)));

            if (format == name::NCHW) {
                m_format = FORMAT_NCHW;
            } else if (format == name::NHWC) {
                m_format = FORMAT_NHWC;
            } else {
                TS_LOG_ERROR << this->op() << " do not support format: " << format << eject;
            }
        }

        int GlobalPooling2D::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto x_tensor = stack[0];

            TS_AUTO_CHECK(x_tensor.dims() == 4);

            if (m_format == FORMAT_NHWC) {
                output.resize(1);
                output[0] = Tensor::Prototype(
                        x_tensor.dtype(),
                        {x_tensor.size(0), 1, 1, x_tensor.size(3)});
            } else {
                TS_LOG_ERROR << "Xnnpack backend only support nhwc layout, but got " << m_format << eject;
            }

            return 1;
        }

        int GlobalPooling2D::run(Stack &stack) {
            std::vector<Tensor::Prototype> output_protos;
            infer(stack, output_protos);

            auto memory_device = running_memory_device();

            Tensor x = stack[0].view(memory_device);

            Tensor out = *stack.push(output_protos[0], memory_device);

            auto &x_shape = x.sizes();
            Size2D ksize;
            static const Padding2D padding(0, 0, 0, 0);
            static const Stride2D stride(1, 1);

            if (m_format == FORMAT_NHWC) {
                ksize = Size2D(x_shape[1], x_shape[2]);
            }

            pooling2d(x, m_type, padding, Padding2DType::BLACK, ksize, stride, m_format, out);

            return 1;
        }

        void GlobalPooling2D::pooling2d(const Tensor &x, Pooling2DType type, Padding2D padding, Padding2DType padding_type,
                                        Size2D ksize, Stride2D stride, Conv2DFormat format, Tensor &out) {
            if (m_type == Pooling2DType::AVG) {
                size_t channels = x.size(3);
                size_t batch_size = x.count() / channels;
                size_t input_stride = channels;
                size_t output_stride = channels;
                if (m_op == nullptr) {
                    float min = -std::numeric_limits<float>::infinity();
                    float max = std::numeric_limits<float>::infinity();
                    m_status = xnn_create_global_average_pooling_nwc_f32(channels, input_stride, output_stride, min, max, 0, &m_op);
                    TS_CHECK_EQ(m_status, xnn_status_success);
                    m_shared_op.reset(m_op, xnn_delete_operator);
                    m_op = m_shared_op.get();
                }
                size_t width = ksize.height * ksize.width;
                m_status = xnn_setup_global_average_pooling_nwc_f32(m_op, batch_size, width, x.data<float>(), out.data<float>(), m_threadpool);
                TS_CHECK_EQ(m_status, xnn_status_success);

                m_status = xnn_run_operator(m_op, m_threadpool);
                TS_CHECK_EQ(m_status, xnn_status_success);

            } else {
                TS_LOG_ERROR << "Not implement global_max_pooling" << eject;

            }
        }
    }
}
using namespace ts;
using namespace xnn;
TS_REGISTER_OPERATOR(GlobalPooling2D, ts::XNNPACK, "xnn::global_pooling2d")
