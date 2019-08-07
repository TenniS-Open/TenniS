#include "compiler/option/pack_translator_option.h"

#include "backend/name.h"
#include "core/tensor_builder.h"
#include "module/menu.h"
#include "kernels/cpu/math_cpu.h"

bool ts::PackTranslatorOption::translate(const ComputingDevice &device, const Node node,
    Node &translated_node, bool output_flag) const {
    auto op_name = node.bubble().op();

    if (Bubble::IsEndPoint(op_name)) {
        if (op_name == Bubble::Parameter)
            translated_node = bubble::param(node.bubble().name());
        else if (op_name == Bubble::Const) {
            translated_node = bubble::bubble(node.bubble());
        }
        return true;
    }

    translated_node = bubble::bubble(node.bubble());

    if (op_name != name::layer::conv2d() && op_name != name::layer::conv2d_v2()
        && op_name != name::layer::inner_prod() || op_name != name::layer::gemm()) {
        Node::Link(translated_node, node.inputs());
        return true;
    }

    auto inputs = node.inputs();
    auto kernel_node = inputs[1];
    if (op_name == name::layer::conv2d() ||op_name == name::layer::inner_prod()) {
        kernel_node = inputs[1];
    }
    else if (op_name == name::layer::conv2d_v2()) {
        kernel_node = inputs[2];
    }

    auto kernel_tensor = kernel_node.bubble().get(name::value);
    auto kernel_shape = kernel_tensor.sizes();
    auto kernel_type = kernel_tensor.dtype();

    int kernel_size_width;
    int kernel_size_height;

    if (op_name == name::layer::conv2d() || op_name == name::layer::conv2d_v2()) {
        kernel_size_height = kernel_shape[0];
        kernel_size_width = kernel_shape[1] * kernel_shape[2] * kernel_shape[3];
    }
    else {
        kernel_size_height = kernel_shape[0];
        kernel_size_width = kernel_shape[1];
    }


    bool need_transpose = false;
    if (node.bubble().has("transpose")) {
        need_transpose = tensor::to_bool(node.bubble().get("transpose"));
    }

    Tensor kernel_packed(kernel_type, kernel_shape);

    if (op_name == name::layer::conv2d() || op_name == name::layer::conv2d_v2()) {
        switch (kernel_type) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
            case DTYPE: { cpu::math<TYPE, TYPE>::pack8_A(kernel_size_height, kernel_size_width, kernel_tensor.data<TYPE>(), kernel_size_width, kernel_packed.data<TYPE>()); break; }
            DECLARE_COMPUTE_RUN(FLOAT32, float);
#undef DECLARE_COMPUTE_RUN
            default: {
                TS_LOG_ERROR << "Pack translator not support data type(" << kernel_type << "): " << type_str(kernel_type) << eject;
                break;
            }
        }
    }
    else {
        switch (kernel_type) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
            case DTYPE: { \
                if(need_transpose){ \
                    Shape transposed_shape({kernel_size_width, kernel_size_height}); \
                    Tensor transposed(kernel_type, transposed_shape); \
                    cpu::math<TYPE, TYPE>::matrix_transpose(kernel_tensor.data<TYPE>(), transposed.data<TYPE>(), kernel_size_height, kernel_size_width); \
                    cpu::math<TYPE, TYPE>::pack8_B(kernel_size_width, kernel_size_height, transposed.data<TYPE>(), kernel_size_height, kernel_packed.data<TYPE>()); \
                } \
                else{ \
                    cpu::math<TYPE, TYPE>::pack8_B(kernel_size_height, kernel_size_width, kernel_tensor.data<TYPE>(), kernel_size_width, kernel_packed.data<TYPE>()); \
                } break;}         
            DECLARE_COMPUTE_RUN(FLOAT32, float);
#undef DECLARE_COMPUTE_RUN
            default: {
                TS_LOG_ERROR << "Pack translator not support data type(" << kernel_type << "): " << type_str(kernel_type) << eject;
                break;
            }
        }
    }


    Node kernel_packed_node = kernel_node;
    kernel_packed_node.bubble().set(name::value, kernel_packed);
    translated_node.bubble().set(name::kernel_need_pack, tensor::from<bool>(false));

    if(op_name == name::layer::conv2d() || op_name == name::layer::inner_prod())
        Node::Link(translated_node, { inputs[0], kernel_packed_node });
    else
        Node::Link(translated_node, { inputs[0], inputs[1], kernel_packed_node });

}


