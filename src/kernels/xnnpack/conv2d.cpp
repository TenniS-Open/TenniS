//
// Created by sen on 2021/9/15.
//
#include "conv2d.h"
#include "backend/base/operator_on_device.h"
#include "kernels/cpu/operator_on_cpu.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "global/operator_factory.h"
#include <valarray>
#include "backend/common_structure.h"
#include "backend/common_function.h"
#include "utils/need.h"
#include "core/tensor_builder.h"
#include "frontend/intime.h"

namespace ts {
    namespace xnn {
        Conv2d::Conv2d() {
            field(name::format, REQUIRED);
            field(name::padding, REQUIRED);
            field(name::padding_value, OPTIONAL, tensor::from(0.0f));
            field(name::stride, REQUIRED);
            field(name::dilation, OPTIONAL);
            field(name::typo::dialations, OPTIONAL);
            field(name::kernel_packed, OPTIONAL, tensor::from<bool>(false));
//            field(name::bias, OPTIONAL);
            field("bias", OPTIONAL, tensor::from(0.0f));
            field("groups", OPTIONAL, tensor::from(1));
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

        void Conv2d::init() {
            supper::init();
            auto ctx = ctx::get<RuntimeContext>();
            m_threadpool = ctx->get_xnn_threadpool();

            if (has("bias")) m_bias = get("bias");
            if (has("groups")) m_groups = tensor::to_int(get("groups"));
            auto format = tensor::to_string(get(name::format));
            auto padding_tensor = tensor::cast(INT32, get(name::padding));
            m_padding_value = tensor::to_float(get(name::padding_value));
            auto stride_tensor = tensor::cast(INT32, get(name::stride));

            if (has(name::kernel_packed)) {
                m_kernel_packed = tensor::to_bool(get(name::kernel_packed));
                TS_CHECK_EQ(m_kernel_packed, false) << eject;
            }

            Tensor dilation_tensor;
            if (has(name::dilation)) {
                dilation_tensor = tensor::cast(INT32, get(name::dilation));
            } else if (has(name::typo::dialations)) {
                dilation_tensor = tensor::cast(INT32, get(name::typo::dialations));
            }

            if (dilation_tensor.empty()) {
                TS_LOG_ERROR << this->op() << " must set " << name::dilation << " or " << name::typo::dialations << eject;
            }

            TS_AUTO_CHECK(padding_tensor.has_shape({4, 2}));
            TS_AUTO_CHECK(stride_tensor.has_shape({4,}));
            TS_AUTO_CHECK(dilation_tensor.has_shape({4,}));

            if (format == name::NHWC) {
                m_format = FORMAT_NHWC;
            } else {
                TS_LOG_ERROR << this->op() << " do not support format: " << format << eject;
            }

            m_padding4x2.resize(8);
            for (size_t i = 0; i < 8; ++i) m_padding4x2[i] = padding_tensor.data<int32_t>(i);
            m_stride4.resize(4);
            for (size_t i = 0; i < 4; ++i) m_stride4[i] = stride_tensor.data<int32_t>(i);
            m_dilation4.resize(4);
            for (size_t i = 0; i < 4; ++i) m_dilation4[i] = dilation_tensor.data<int32_t>(i);

            if (m_padding4x2[0] != 0 ||
                m_padding4x2[1] != 0 ||
                m_padding4x2[2] != 0 ||
                m_padding4x2[3] != 0) {
                TS_LOG_ERROR << this->op() << " do not support padding: " << to_string(m_padding4x2) << eject;
            }
            if (m_stride4[0] != 1 ||
                m_stride4[1] != 1) {
                TS_LOG_ERROR << this->op() << " do not support stride: " << to_string(m_stride4) << eject;
            }
            if (m_dilation4[0] != 1 ||
                m_dilation4[1] != 1) {
                TS_LOG_ERROR << this->op() << " do not support dialations: " << to_string(m_dilation4) << eject;
            }

        }

        int Conv2d::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
//            auto cpu_conv2d_infer = OperatorCreator::Create(memory_device().type(), name::layer::conv2d(), true);
//            InferOperator(cpu_conv2d_infer, stack, 2, output);
//            return 1;
            TS_AUTO_CHECK(stack.size() == 2);
            if (m_format == FORMAT_NCHW) {
                TS_LOG_ERROR << "Only support NHWC layout in xnnpack backend." << eject;
            }

            auto x_tensor = stack[0];
            auto w_tensor = stack[1];

            TS_AUTO_CHECK(x_tensor.dims() == 4);
            TS_AUTO_CHECK(w_tensor.dims() == 4);

            TS_AUTO_CHECK(x_tensor.dtype() == w_tensor.dtype());

            if(w_tensor.size(3) != x_tensor.size(3)) {
                TS_LOG_ERROR << "Conv2d assert failed when x=" << x_tensor.proto() << ", w=" << w_tensor.proto() << eject;
            }

            Size2D x;
            Size2D ksize;
            Padding2D padding;
            Stride2D stride;
            Dilation2D dialations;

            x = Size2D(x_tensor.size(1), x_tensor.size(2));
            ksize = Size2D(w_tensor.size(1), w_tensor.size(2));
            padding = Padding2D(m_padding4x2[4], m_padding4x2[5], m_padding4x2[6], m_padding4x2[7]);
            stride = Stride2D(m_stride4[2], m_stride4[3]);
            dialations = Stride2D(m_dilation4[2], m_dilation4[3]);

            Size2D y = conv2d_forward(x, padding, ksize, stride, dialations);

            Tensor::Prototype out_proto;
            out_proto = Tensor::Prototype(x_tensor.dtype(), {x_tensor.size(0), y.height, y.width, w_tensor.size(0)});

            output.resize(1);
            output[0] = out_proto;
            return 1;
        }

        int Conv2d::run(Stack &stack) {
            std::vector<Tensor::Prototype> output_protos;
            infer(stack, output_protos);

            auto memory_device = running_memory_device();
            Tensor x = stack[0].view(memory_device);
            Tensor w = stack[1].view(memory_device);

            Tensor out = *stack.push(output_protos[0], memory_device);

            Padding2D padding;
            Stride2D stride;
            Dilation2D dilation;

            padding = Padding2D(m_padding4x2[4], m_padding4x2[5], m_padding4x2[6], m_padding4x2[7]);
            stride = Stride2D(m_stride4[2], m_stride4[3]);
            dilation = Stride2D(m_dilation4[2], m_dilation4[3]);

            conv2d(x, padding, m_padding_value, w, stride, dilation, m_format, out);

            return 1;
        }

        void Conv2d::conv2d(const Tensor &x, const Padding2D &padding, float padding_value, const Tensor &w,
                            const Stride2D &stride, const Dilation2D &dilation, Conv2DFormat format, Tensor &out) {
            // only support zero padding
            TS_CHECK(padding_value == 0);

            if (m_op == nullptr) {
                size_t groups = m_groups;
                size_t group_input_channels = x.size(3) / groups;;
                size_t group_output_channels = out.size(3) / groups;;

                size_t input_channel_stride = groups * group_input_channels;
                size_t output_channel_stride = groups * group_output_channels;
                float min = -std::numeric_limits<float>::infinity();
                float max = std::numeric_limits<float>::infinity();
                m_status = xnn_create_convolution2d_nhwc_f32(padding.top, padding.right, padding.bottom, padding.left,
                                                             w.size(1), w.size(2),
                                                             stride.height, stride.width, dilation.height,
                                                             dilation.width, groups, group_input_channels,
                                                             group_output_channels,
                                                             input_channel_stride, output_channel_stride,
                                                             w.data<float>(), m_bias.data<float>(), min, max, 0, &m_op);
                TS_CHECK_EQ(m_status, xnn_status_success);
                m_shared_op.reset(m_op, xnn_delete_operator);
                m_op = m_shared_op.get();
            }

            int batch_size = x.size(0);
            int input_height = x.size(1);
            int input_width = x.size(2);
            m_status = xnn_setup_convolution2d_nhwc_f32(m_op, batch_size, input_height, input_width, x.data<float>(), out.data<float>(), m_threadpool);
            TS_CHECK(m_status == xnn_status_success);

            m_status = xnn_run_operator(m_op, m_threadpool);
            TS_CHECK(m_status == xnn_status_success);
        }

    }
}

using namespace ts;
using namespace xnn;
TS_REGISTER_OPERATOR(Conv2d, ts::XNNPACK, "xnn::conv2d")