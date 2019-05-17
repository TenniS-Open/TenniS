#include <backend/base/base_conv2d_winograd.h>

#include "backend/base/base_conv2d_winograd.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

#include "backend/common_function.h"
#include "utils/need.h"

namespace ts {
    namespace base{
        Conv2DWinograd::Conv2DWinograd() {
            field(name::winograd_model, OPTIONAL, tensor::from(name::winograd_f63));
            field(name::format, REQUIRED);
            //field(name::padding, REQUIRED);
            //field(name::padding_value, OPTIONAL, tensor::from(0.0f));
        }

        void Conv2DWinograd::init() {
            supper::init();

            auto format = tensor::to_string(get(name::format));
            //auto padding_tensor = tensor::cast(INT32, get(name::padding));
           // m_padding_value = tensor::to_float(get(name::padding_value));

            //TS_AUTO_CHECK(padding_tensor.has_shape({ 4, 2 }));

            if (format == name::NCHW) {
                m_format = FORMAT_NCHW;
            }
            else if (format == name::NHWC) {
                m_format = FORMAT_NHWC;
            }
            else {
                TS_LOG_ERROR << this->op() << " do not support format: " << format << eject;
            }

            //m_padding4x2.resize(8);
            //for (size_t i = 0; i < 8; ++i) m_padding4x2[i] = padding_tensor.data<int32_t>(i);

            //// only support native conv2d
            //if (m_format == FORMAT_NCHW) {
            //    if (m_padding4x2[0] != 0 ||
            //        m_padding4x2[1] != 0 ||
            //        m_padding4x2[2] != 0 ||
            //        m_padding4x2[3] != 0) {
            //        TS_LOG_ERROR << this->op() << " do not support padding: " << to_string(m_padding4x2) << eject;
            //    }
            //}
            //else if (m_format == FORMAT_NHWC) {
            //    if (m_padding4x2[0] != 0 ||
            //        m_padding4x2[1] != 0 ||
            //        m_padding4x2[6] != 0 ||
            //        m_padding4x2[7] != 0) {
            //        TS_LOG_ERROR << this->op() << " do not support padding: " << to_string(m_padding4x2) << eject;
            //    }
            //}

            auto winograd_model = tensor::to_string(get(name::winograd_model));
            if (winograd_model == name::winograd_f63) {
                m_winograd_model = F6X6_3X3;
            }
            else if (winograd_model == name::winograd_f23) {
                m_winograd_model = F2X2_3X3;
            }
            else {
                TS_LOG_ERROR << this->op() << " do not support winograd model: " << winograd_model << eject;
            }
        }

        int Conv2DWinograd::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto x_tensor = stack[0];
            auto k_trans_tensor = stack[1];

            TS_AUTO_CHECK(x_tensor.dims() == 4);
            TS_AUTO_CHECK(k_trans_tensor.dims() == 4);

            Size2D x;
            //winograd conv2d only suppot ksize == 3x3 && stride == 1 && dialations == 1
            Size2D ksize(3, 3);
            Stride2D stride(1, 1);
            Dilation2D dialations(1, 1);
            Padding2D padding(0, 0, 0, 0);
            //Padding2D padding;
            if (m_format == FORMAT_NCHW) {
                x = Size2D(x_tensor.size(2), x_tensor.size(3));
                //padding = Padding2D(m_padding4x2[4], m_padding4x2[5], m_padding4x2[6], m_padding4x2[7]);
            }
            else if (m_format == FORMAT_NHWC) {
                x = Size2D(x_tensor.size(1), x_tensor.size(2));
                //padding = Padding2D(m_padding4x2[2], m_padding4x2[3], m_padding4x2[4], m_padding4x2[5]);
            }

            Size2D y = conv2d_forward(x, padding, ksize, stride, dialations);

            Tensor::Prototype out_proto;

            if (m_format == FORMAT_NCHW) {
                out_proto = Tensor::Prototype(
                    x_tensor.dtype(),
                    { x_tensor.size(0), k_trans_tensor.size(0), y.height, y.width });
            }
            else if (m_format == FORMAT_NHWC) {
                out_proto = Tensor::Prototype(
                    x_tensor.dtype(),
                    { x_tensor.size(0), y.height, y.width, k_trans_tensor.size(0) });
            }

            output.resize(1);
            output[0] = out_proto;

            return 1;
        }

        int Conv2DWinograd::run(ts::Stack &stack) {
            std::vector<Tensor::Prototype> output;
            infer(stack, output);

            auto memory_device = running_memory_device();

            auto x_tensor = stack[0].view(memory_device);
            auto kernel_tensor = stack[1].view(memory_device);

            auto out = *stack.push(output[0], memory_device);

            //Padding2D padding;

            //if (m_format == FORMAT_NCHW) {
            //    padding = Padding2D(m_padding4x2[4], m_padding4x2[5], m_padding4x2[6], m_padding4x2[7]);
            //}
            //else if (m_format == FORMAT_NHWC) {
            //    padding = Padding2D(m_padding4x2[2], m_padding4x2[3], m_padding4x2[4], m_padding4x2[5]);
            //}

            {
                stack.push_base(3); // empty base
                need pop_base(&Stack::pop_base, &stack);

                TS_AUTO_CHECK(stack.size() == 0);

                //conv2d_winograd(x_tensor, m_winograd_model, padding, m_padding_value, kernel_tensor, m_format, out, stack);
                conv2d_winograd(x_tensor, m_winograd_model, kernel_tensor, m_format, out, stack);

                stack.clear();
            }

            return 1;
        }
    }
}