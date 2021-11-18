//
// Created by sen on 2021/9/27.
//
#include <runtime/workbench.h>
#include "xnnpack_translator_option.h"
#include "backend/name.h"
#include "module/menu.h"


bool ts::XnnpackTranslatorOption::translate(const ComputingDevice &device, const Node node, Node &translated_node,
                                            const std::string &params, bool output_flag) const {
    std::string xnn_op_prefix = "xnn::";

    auto op_name = node.bubble().op();

    if (Bubble::IsEndPoint(op_name)) {
        if (op_name == Bubble::Parameter)
            translated_node = bubble::param(node.bubble().name());
        else if (op_name == Bubble::Const) {
            translated_node = bubble::bubble(node.bubble());
        }
        return true;
    }

    if (op_name == name::layer::add_bias()) {
        auto _input = node.inputs()[0];
        auto _bias = node.inputs()[1];
        if (_bias->op() == Bubble::Const && _input->op() == xnn_op_prefix + name::layer::conv2d()) {
            _input.bubble().set("bias", _bias->get(name::value));
            translated_node = bubble::bubble(_input.bubble());
            Node::Link(translated_node, _input.inputs());
            return true;
        }
    }

    translated_node = bubble::bubble(node.bubble());

    if (op_name == name::layer::concat() && !node->has("#translated")) {
        bool need_translation = false;
        // prior nodes layout whether nhwc
        for (auto &node_in : node.inputs()) {
            if (node_in->op() == name::layer::copy()) node_in = node_in.input(0);
            if (node_in->name().find("_nchw2nhwc") != std::string::npos || node_in->op().find(xnn_op_prefix) != std::string::npos) {
                need_translation = true;
            }
        }

        if (need_translation) {
            int old_dim = tensor::to_int(translated_node->get(name::dim));
            int new_dim = old_dim;
            old_dim = old_dim < 0 ? old_dim + 4 : old_dim;
            switch (old_dim) {
                case 0: new_dim = 0; break;
                case 1: new_dim = 3; break;
                case 2: new_dim = 1; break;
                case 3: new_dim = 2; break;
                default: new_dim = old_dim; break;
            }
            translated_node->set("#translated", tensor::from({true}));
            translated_node->set(name::dim, tensor::from({new_dim}));
            Node::Link(translated_node, node.inputs());
            return true;
        }
    }

    if (op_name != xnn_op_prefix + name::layer::conv2d()) {
        Node::Link(translated_node, node.inputs());
        return true;
    }

//    if (op_name != name::layer::conv2d() && op_name != name::layer::conv2d_v2()
//        && op_name != name::layer::inner_prod() && op_name != name::layer::gemm()) {
//        Node::Link(translated_node, node.inputs());
//        return true;
//    }

    // kernel should be nhwc layout
    auto inputs = node.inputs();
    auto kernel_node = inputs[1];
    if (op_name == xnn_op_prefix + name::layer::conv2d()) {
        kernel_node = inputs[1];
    }
    else if (op_name == xnn_op_prefix + name::layer::conv2d_v2()) {
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

    if (op_name == xnn_op_prefix + name::layer::conv2d() || op_name == xnn_op_prefix + name::layer::conv2d_v2()) {
        // transpose kernel from nchw to nhwc
        auto op_transpose = OperatorCreator::Create(CPU, name::layer::transpose(), true);
        TS_CHECK_NQ(op_transpose, nullptr) << "Can not find operator: " << name::layer::transpose() << eject;

        op_transpose->set(Bubble::RetentionParam::op, tensor::from(name::layer::transpose()));
        op_transpose->set(Bubble::RetentionParam::name, tensor::from("_core" + name::layer::transpose()));
        op_transpose->set(name::permute, tensor::from({0, 2, 3, 1}));
//        op_transpose->set(name::permute, tensor::from({0, 3, 1, 2}));

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

    if(op_name == xnn_op_prefix + name::layer::conv2d())
        Node::Link(translated_node, { inputs[0], kernel_transposed_node });
    else
        Node::Link(translated_node, { inputs[0], inputs[1], kernel_transposed_node });

    return true;
}
