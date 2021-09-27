//
// Created by sen on 2021/9/27.
//
#include <runtime/workbench.h>
#include "compiler/option/layout_translator_option.h"
#include "backend/name.h"
#include "module/menu.h"


bool ts::LayoutTranslatorOption::translate(const ComputingDevice &device, const Node node, Node &translated_node,
                                           const std::string &params, bool output_flag) const {
    bool defined_xnnpack = false;
#ifdef TS_USE_XNNPACK
    defined_xnnpack = true;
#endif
    if (!(device.type() == ts::CPU && defined_xnnpack)) return true;
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
        && op_name != name::layer::inner_prod() && op_name != name::layer::gemm()) {
        Node::Link(translated_node, node.inputs());
        return true;
    }

    // kernel should be nhwc layout
    auto inputs = node.inputs();
    auto kernel_node = inputs[1];
    if (op_name == name::layer::conv2d()) {
        kernel_node = inputs[1];
    }
    else if (op_name == name::layer::conv2d_v2()) {
        kernel_node = inputs[2];
    }

    if (kernel_node->op() != Bubble::Const) {
        Node::Link(translated_node, node.inputs());
        return true;
    }

    auto kernel_tensor = kernel_node.bubble().get(name::value);
    auto kernel_shape = kernel_tensor.sizes();
    auto kernel_type = kernel_tensor.dtype();

    Tensor kernel_nhwc(kernel_type, {kernel_shape[0], kernel_shape[2], kernel_shape[3], kernel_shape[1]});

    if (op_name == name::layer::conv2d() || op_name == name::layer::conv2d_v2()) {
        // transpose kernel from nchw to nhwc
        auto op_transpose = OperatorCreator::Create(CPU, name::layer::transpose(), true);
        TS_CHECK_NQ(op_transpose, nullptr) << "Can not find operator: " << name::layer::transpose() << eject;

        op_transpose->set(Bubble::RetentionParam::op, tensor::from(name::layer::transpose()));
        op_transpose->set(Bubble::RetentionParam::name, tensor::from("_core" + name::layer::transpose()));
        op_transpose->set(name::permute, tensor::from({0, 2, 3, 1}));

        op_transpose->init();

        std::shared_ptr<Workbench> bench_ptr;
        bench_ptr = std::make_shared<Workbench>(device);

        Workbench &bench = *bench_ptr;
        std::vector<Tensor> run_output;
        bench.offline_run(op_transpose, {kernel_tensor}, run_output);
        kernel_nhwc = run_output[0];
    }

    Node kernel_transposed_node = kernel_node;
    kernel_transposed_node.bubble().set(name::value, kernel_nhwc);
    translated_node.bubble().set(name::format, tensor::from(name::NHWC));

    if(op_name == name::layer::conv2d())
        Node::Link(translated_node, { inputs[0], kernel_transposed_node });
    else
        Node::Link(translated_node, { inputs[0], inputs[1], kernel_transposed_node });

    return true;
}

TS_REGISTER_TRANSLATOR_OPTION(ts::LayoutTranslatorOption)
