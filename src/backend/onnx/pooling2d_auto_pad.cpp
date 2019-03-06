//
// Created by kier on 2019/1/12.
//

#include "backend/onnx/pooling2d_auto_pad.h"

#include "backend/name.h"
#include "core/tensor_builder.h"
#include "utils/assert.h"

#include "runtime/stack.h"

#include "backend/common_function.h"

#include "core/device.h"
#include "global/operator_factory.h"

namespace ts {
    namespace onnx {

        Pooling2DAutoPad::Pooling2DAutoPad() {
            this->field(name::auto_pad, OPTIONAL, tensor::from(name::NOTSET));
            this->field(name::padding, OPTIONAL, tensor::build(INT32, {4, 2}, {0, 0, 0, 0, 0, 0, 0, 0}));
        }

        void Pooling2DAutoPad::init() {
            supper::init();

            auto auto_pad_str = tensor::to_string(this->get(name::auto_pad));

            if (auto_pad_str == name::NOTSET) {
                this->auto_pad = AutoPadType::NOTSET;
            } else if (auto_pad_str == name::SAME_LOWER) {
                this->auto_pad = AutoPadType::SAME_LOWER;
            } else if (auto_pad_str == name::SAME_UPPER) {
                this->auto_pad = AutoPadType::SAME_UPPER;
            } else if (auto_pad_str == name::VALID) {
                this->auto_pad = AutoPadType::VALID;
            } else {
                TS_LOG_ERROR << "Not supported auto_pad=" << auto_pad_str << eject;
            }

            auto static_padding = tensor::cast(INT32, this->get(name::padding));

            TS_AUTO_CHECK(static_padding.has_shape({4, 2}));

            // IS NCHW format
            this->static_padding.top = static_padding.data<int32_t>()[4];
            this->static_padding.bottom = static_padding.data<int32_t>()[5];
            this->static_padding.left = static_padding.data<int32_t>()[6];
            this->static_padding.right = static_padding.data<int32_t>()[7];
        }

        int Pooling2DAutoPad::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            output.resize(1);
            output[0] = Tensor::Prototype(INT32, {4, 2});
            return 1;
        }

        static Size2D valid_pooling2d_forward(const Size2D &x, const Padding2D &padding, const KSize2D &ksize,
                                        const Stride2D &stride) {
            Size2D y;
            y.height = int(std::floor((x.height + padding.top + padding.bottom - ksize.height) / (float)stride.height + 1));
            y.width = int(std::floor((x.width + padding.left + padding.right - ksize.width) / (float)stride.width + 1));
            return y;
        }

        int Pooling2DAutoPad::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 3);

            // get output
            auto &dynamic_padding_tensor = *stack.push(INT32, {4, 2}, MemoryDevice(CPU));

            // convert input
            auto shape = stack[0].sizes();
            auto ksize_tensor = tensor::cast(INT32, stack[1]);
            auto stride_tensor = tensor::cast(INT32, stack[2]);

            Padding2D dynamic_padding;

            KSize2D ksize;
            Stride2D stride;
            Size2D input_size;

            ksize.height = ksize_tensor.data<int32_t>()[2];
            ksize.width = ksize_tensor.data<int32_t>()[3];
            stride.height = stride_tensor.data<int32_t>()[2];
            stride.width = stride_tensor.data<int32_t>()[3];
            input_size.height = shape[2];
            input_size.width = shape[3];

            TS_LOG_ERROR << "Not implement." << eject;

            Size2D expected_output_size = valid_pooling2d_forward(input_size, static_padding, ksize, stride);
            Size2D expected_input_size = pooling2d_backward(expected_output_size, static_padding, ksize, stride);
            dynamic_padding.top = static_padding.top;
            dynamic_padding.left = static_padding.left;
            dynamic_padding.bottom = static_padding.bottom + expected_input_size.height - input_size.height;
            dynamic_padding.right = static_padding.right + expected_input_size.width - input_size.width;

            auto dynamic_padding_data = dynamic_padding_tensor.data<int32_t>();

            dynamic_padding_data[0] = 0;
            dynamic_padding_data[1] = 0;
            dynamic_padding_data[2] = 0;
            dynamic_padding_data[3] = 0;
            dynamic_padding_data[4] = dynamic_padding.top;
            dynamic_padding_data[5] = dynamic_padding.bottom;
            dynamic_padding_data[6] = dynamic_padding.left;
            dynamic_padding_data[7] = dynamic_padding.right;

            return 1;
        }
    }
}

using namespace ts;
using namespace ts::onnx;

TS_REGISTER_OPERATOR(Pooling2DAutoPad, CPU, name::layer::onnx_pooling2d_padding())
