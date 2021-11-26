//
// Created by sen on 2021/9/27.
//
#include <runtime/workbench.h>
#include <kernels/common/math.h>
#include "xnnpack_converter_option.h"
#include "xnnpack_translator_option.h"
#include "backend/name.h"
#include "module/menu.h"

static ts::Node translate_conv_family(const ts::ComputingDevice &device, ts::Node &node) {
    std::string xnn_op_prefix = "xnn::";
    auto op_name = node.bubble().op();

    auto inputs = node.inputs();
    auto kernel_node = inputs[1];
    if (op_name == xnn_op_prefix + ts::name::layer::conv2d()
        || op_name == xnn_op_prefix + ts::name::layer::depthwise_conv2d()) {
        kernel_node = inputs[1];
    }
    else if (op_name == xnn_op_prefix + ts::name::layer::conv2d_v2()
            || op_name == xnn_op_prefix + ts::name::layer::depthwise_conv2d_v2()) {
        kernel_node = inputs[2];
    }

    auto translated_node = ts::bubble::bubble(node.bubble());

    auto kernel_tensor = kernel_node.bubble().get(ts::name::value);
    auto kernel_shape = kernel_tensor.sizes();
    auto kernel_type = kernel_tensor.dtype();

    ts::Tensor kernel_nhwc(kernel_type, {kernel_shape[0], kernel_shape[2], kernel_shape[3], kernel_shape[1]});

    if (op_name == xnn_op_prefix + ts::name::layer::conv2d()
        || op_name == xnn_op_prefix + ts::name::layer::conv2d_v2()
        || op_name == xnn_op_prefix + ts::name::layer::depthwise_conv2d()
        || op_name == xnn_op_prefix + ts::name::layer::depthwise_conv2d_v2()) {
        // transpose kernel from nchw to nhwc
        auto op_transpose = ts::OperatorCreator::Create(ts::CPU, ts::name::layer::transpose(), true);
        TS_CHECK_NQ(op_transpose, nullptr) << "Can not find operator: " << ts::name::layer::transpose() << ts::eject;

        op_transpose->set(ts::Bubble::RetentionParam::op, ts::tensor::from(ts::name::layer::transpose()));
        op_transpose->set(ts::Bubble::RetentionParam::name, ts::tensor::from("_core" + ts::name::layer::transpose()));

        op_transpose->set(ts::name::permute, ts::tensor::from({0, 2, 3, 1}));

        op_transpose->init();

        std::shared_ptr<ts::Workbench> bench_ptr;
        bench_ptr = std::make_shared<ts::Workbench>(device);

        ts::Workbench &bench = *bench_ptr;
        std::vector<ts::Tensor> run_output;
        bench.offline_run(op_transpose, {kernel_tensor}, run_output);
        kernel_nhwc = run_output[0];
    }

    ts::Node kernel_transposed_node = kernel_node;
    kernel_transposed_node.bubble().set(ts::name::value, kernel_nhwc);
    kernel_transposed_node.bubble().set(ts::name::format, ts::tensor::from(ts::name::NHWC));

    if(op_name == xnn_op_prefix + ts::name::layer::conv2d() || op_name == xnn_op_prefix + ts::name::layer::depthwise_conv2d())
        ts::Node::Link(translated_node, { inputs[0], kernel_transposed_node });
    else
        ts::Node::Link(translated_node, { inputs[0], inputs[1], kernel_transposed_node });
    return translated_node;
}

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
        if (_bias->op() == Bubble::Const
            && (_input->op() == xnn_op_prefix + name::layer::conv2d()
                || _input->op() == xnn_op_prefix + name::layer::conv2d_v2()
                || _input->op() == xnn_op_prefix + name::layer::depthwise_conv2d()
                || _input->op() == xnn_op_prefix + name::layer::depthwise_conv2d_v2())) {
            _input.bubble().set("bias", _bias->get(name::value));
            auto kernel_translated_node = translate_conv_family(device, _input);
            translated_node = bubble::bubble(kernel_translated_node.bubble());
            translated_node->set(name::format, ts::tensor::from(name::NHWC));
            Node::Link(translated_node, _input.inputs());
            return true;
        }
    }

    translated_node = bubble::bubble(node.bubble());

//    if (op_name == name::layer::concat() && !node->has("#translated")) {
//        bool need_translation = false;
//        // prior nodes layout whether nhwc
//        for (auto &node_in : node.inputs()) {
//            auto node_in_op_name = node_in->op();
//            if (node_in_op_name.find(xnn_op_prefix) != std::string::npos) {
//                node_in_op_name = node_in_op_name.substr(xnn_op_prefix.length(), node_in_op_name.length());
//            }
//            if (xnn::xnn_support_op.find(node_in_op_name) != xnn::xnn_support_op.end() ||
//                (xnn::xnn_route_op.find(node_in_op_name) != xnn::xnn_route_op.end() && node_in_op_name != name::layer::concat())) {
//                need_translation = true;
//            }
//        }
//
//        if (need_translation) {
//            int old_dim = tensor::to_int(translated_node->get(name::dim));
//            int new_dim = old_dim;
//            old_dim = old_dim < 0 ? old_dim + 4 : old_dim;
//            switch (old_dim) {
//                case 0: new_dim = 0; break;
//                case 1: new_dim = 3; break;
//                case 2: new_dim = 1; break;
//                case 3: new_dim = 2; break;
//                default: new_dim = old_dim; break;
//            }
//            translated_node->set("#translated", tensor::from({true}));
//            translated_node->set(name::dim, tensor::from({new_dim}));
//            Node::Link(translated_node, node.inputs());
//            return true;
//        }
//    }

    if (op_name == xnn_op_prefix + name::layer::global_pooling2d()) {
        translated_node->set(name::format, tensor::from(name::NHWC));
        Node::Link(translated_node, node.inputs());
        return true;
    }

    if (op_name == xnn_op_prefix + name::layer::gemm()) {
        // if satisfy situation, then translate to xnn::inner_prod
        // else translate to cpu::gemm
        float alpha = tensor::to_float(node->get(name::alpha));
        float beta = tensor::to_float(node->get(name::beta));
        bool transA = tensor::to_bool(node->get(name::transA));
        bool transB = tensor::to_bool(node->get(name::transB));
        bool xnn_inner_prod_flag = transB && !transA && ts::near(alpha, float(1)) && ts::near(beta, float(1));
        if (xnn_inner_prod_flag) {
            translated_node->op(xnn_op_prefix + name::layer::inner_prod());
            Node::Link(translated_node, node.inputs());
            return true;
        } else {
            translated_node->op(name::layer::gemm());
            Node::Link(translated_node, node.inputs());
            return true;
        }
    }

    if (op_name != xnn_op_prefix + name::layer::conv2d() && op_name != xnn_op_prefix + name::layer::depthwise_conv2d()) {
        Node::Link(translated_node, node.inputs());
        return true;
    }

//    if (op_name != name::layer::conv2d() && op_name != name::layer::conv2d_v2()
//        && op_name != name::layer::inner_prod() && op_name != name::layer::gemm()) {
//        Node::Link(translated_node, node.inputs());
//        return true;
//    }

    // no bias convolution family case
    // kernel should be nhwc layout
    auto inputs = node.inputs();
    auto kernel_node = inputs[1];
    if (op_name == xnn_op_prefix + name::layer::conv2d() || op_name == xnn_op_prefix + name::layer::depthwise_conv2d()) {
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

    if (op_name == xnn_op_prefix + name::layer::conv2d()
        || op_name == xnn_op_prefix + name::layer::conv2d_v2()
        || op_name == xnn_op_prefix + name::layer::depthwise_conv2d()
        || op_name == xnn_op_prefix + name::layer::depthwise_conv2d_v2()) {
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

    if(op_name == xnn_op_prefix + name::layer::conv2d() || op_name == xnn_op_prefix + name::layer::depthwise_conv2d())
        Node::Link(translated_node, { inputs[0], kernel_transposed_node });
    else
        Node::Link(translated_node, { inputs[0], inputs[1], kernel_transposed_node });

    return true;
}
