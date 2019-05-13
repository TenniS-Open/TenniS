//
// Created by kier on 19-5-7.
//

#include <compiler/option/zipper_option.h>
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "module/menu.h"

#include <valarray>

namespace ts {
    static std::vector<const ZipperOption*> &GetStaticOptions() {
        static std::vector<const ZipperOption*> options;
        return options;
    }

    const std::vector<const ZipperOption *> &GetFullOptions() {
        return GetStaticOptions();
    }

    void RegisterOption(const ZipperOption *option) {
        auto &options = GetStaticOptions();
        options.emplace_back(option);
    }

    class EmptyZipperOption : public ZipperOption {
    public:
        bool zip(const ComputingDevice &device, Node node, Node &zipped_node) const final {
            return false;
        }
    };

    class Conv2dZipperOption : public ZipperOption {
    public:
        bool zip(const ComputingDevice &device, Node node, Node &zipped_node) const final {
            if (device.type() != CPU) 
                return false;

            auto bubble = node.bubble();
            auto op_name = bubble.op();
            if (op_name != name::layer::conv2d() && op_name != name::layer::conv2d_v2())
                return false;
            
            auto format_tensor = bubble.get(name::format);
            auto format = tensor::to_string(format_tensor);
            auto stride_tensor = tensor::cast(INT32, bubble.get(name::stride));

            if (bubble.has(name::dilation)) {
                auto dilation_tensor = tensor::cast(INT32, bubble.get(name::dilation));
                std::valarray<int> dilation4;
                dilation4.resize(4);
                for (size_t i = 0; i < 4; ++i)
                    dilation4[i] = dilation_tensor.data<int32_t>(i);
                if (format == name::NCHW) {
                    if (dilation4[2] != 1 || dilation4[3] != 1)
                        return false;
                }
                else if (format == name::NHWC) {
                    if (dilation4[1] != 1 || dilation4[2] != 1)
                        return false;
                }
                else {
                    TS_LOG_ERROR << "Conv2d do not support format: " << format << eject;
                }
            }

            std::valarray<int> stride4;
            stride4.resize(4);
            for (size_t i = 0; i < 4; ++i) 
                stride4[i] = stride_tensor.data<int32_t>(i);

            auto inputs = node.inputs();
            //Size2D ksize; KSize2D stride_size; Dilation2D dilation_size
            if (format == name::NCHW) {
                if (stride4[2] != 1 || stride4[3] != 1)
                    return false;
            }
            else if (format == name::NHWC) {
                if (stride4[1] != 1 || stride4[2] != 1)
                    return false;
            }
            else {
                TS_LOG_ERROR << "Conv2d do not support format: " << format << eject;
            }

            Tensor kernel_tensor;
            if (op_name == name::layer::conv2d()) {
                TS_AUTO_CHECK(inputs.size() == 2);
                kernel_tensor = inputs[1].bubble().get(name::value);
                auto kernel_shape = kernel_tensor.sizes();
                if (format == name::NCHW) {
                    if (kernel_shape[2] != 3 || kernel_shape[3] != 3)
                        return false;
                }
                else {
                    if (kernel_shape[1] != 3 || kernel_shape[2] != 3)
                        return false;
                }
            }

            if (op_name == name::layer::conv2d()) {  
                auto padding_param_tensor = node.bubble().get(name::padding);
                std::string pad_param_name = node.bubble().name() + "_pad_param";
                auto padding = bubble::data(pad_param_name, padding_param_tensor,CPU);
                std::string pad_name = node.bubble().name() + "_pad";
                auto pad_node = bubble::op(pad_name, name::layer::pad(), { inputs[0],padding });
                if (node.bubble().has(name::padding_value)) {
                    auto padding_value_tensor = node.bubble().get(name::padding_value);
                    pad_node.bubble().set(name::padding_value, padding_value_tensor);
                }
         
                std::string transform_kernel_name = node.bubble().name() + "_transform_kernel";
                auto transform_kernel_node = bubble::op(transform_kernel_name, name::layer::winograd_transform_kernel(), { inputs[1] });

                std::string winograd_name = node.bubble().name() + "_conv2d_winograd";
                zipped_node = bubble::op(winograd_name, name::layer::conv2d_winograd(), { pad_node,transform_kernel_node });
            }
            else {
                std::string pad_name = node.bubble().name() + "_pad";
                auto pad_node = bubble::op(pad_name, name::layer::pad(), { inputs[0], inputs[1] });
                auto padding_value_tensor = inputs[1].bubble().get(name::value);
                pad_node.bubble().set(name::padding_value, padding_value_tensor);

                std::string transform_kernel_name = node.bubble().name() + "_transform_kernel";
                auto transform_kernel_node = bubble::op(transform_kernel_name, name::layer::winograd_transform_kernel(), { inputs[2] });

                std::string winograd_name = node.bubble().name() + "_conv2d_winograd";
                zipped_node = bubble::op(winograd_name, name::layer::conv2d_winograd(), { pad_node,transform_kernel_node });
            }

            zipped_node.bubble().set(name::format, format_tensor);

            return true;
        }
    };
}

TS_REGISTER_ZIPPER_OPTION(ts::EmptyZipperOption)
TS_REGISTER_ZIPPER_OPTION(ts::Conv2dZipperOption)