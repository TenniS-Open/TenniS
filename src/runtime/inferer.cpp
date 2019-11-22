//
// Created by kier on 2019/11/20.
//

#include <core/tensor_builder.h>
#include "runtime/inferer.h"
#include "global/shape_inferer_factory.h"

#include "runtime/stack.h"
#include "global/operator_factory.h"

namespace ts {
    static Tensor get_value(const Node &node) {
        if (node->op() == Bubble::Const) {
            return node->get("value");
        }
        if (node->has("#value")) {
            return node->get("#value");
        }
        return Tensor();
    }

    static void try_run(Node &node) {
        std::vector<Tensor> inputs;
        for (auto &i : node.inputs()) {
            inputs.push_back(get_value(i));
            if (inputs.back().empty()) return;
        }
        MemoryDevice memory_device(CPU);
        Stack stack(memory_device);

        auto op = OperatorCreator::CreateNoException(memory_device.type(), node->op());
        if (op == nullptr) return;

        Tensor output;

        try {
            for (const auto &it : node->params()) {
                op->set(it.first, it.second);
            }
            op->init();
            for (auto &t : inputs) {
                stack.push(t);
            }
            auto out = op->run(stack);
            stack.erase(0, -out);
            if (out == 1) {
                output = stack[0];
            } else {
                output = Tensor::Pack(std::vector<Tensor>(stack.begin(), stack.end()));
            }
        } catch (...) {
            return;
        }

        node->set("#value", output);
    }

    static TensorPrototype update_cache(Node &node, std::unordered_map<Node, TensorPrototype> &cache, const TensorPrototype &proto) {
        cache.insert(std::make_pair(node, proto));
        if (node->op() != Bubble::Parameter) {
            auto N = proto.fields_count();
            for (size_t i =0; i < N; ++i) {
                auto shape_key = Bubble::RetentionParam::shape + (i ? "_" + std::to_string(i) : "");
                auto dtype_key = Bubble::RetentionParam::dtype + (i ? "_" + std::to_string(i) : "");
                auto dtype_shape = proto.field(i);
                node->set(shape_key, tensor::build(INT32, dtype_shape.sizes()));
                node->set(dtype_key, tensor::build(INT32, int32_t(dtype_shape.dtype())));
            }
        }
        return proto;
    }

    TensorPrototype infer(Node &node, std::unordered_map<Node, TensorPrototype> &cache) {
        auto cache_it = cache.find(node);
        if (cache_it != cache.end()) return cache_it->second;

#define RETURN_CACHE(proto) return update_cache(node, cache, proto)

        std::vector<TensorPrototype> input_proto;
        for (auto &i : node.inputs()) {
            input_proto.emplace_back(infer(i, cache));
            if (input_proto.back().dtype() == VOID) {
                RETURN_CACHE(VOID);
            }
        }

        auto shape_infer = ShapeInferer::Query(node->op());
        if (shape_infer == nullptr) {
            TS_LOG_ERROR << "No method to infer " << node->op() << ":" << node->name();
            RETURN_CACHE(VOID);
        }

        auto output_proto = shape_infer(node, input_proto);
        if (output_proto.dtype() == VOID) {
            TS_LOG_ERROR << "Failed to infer " << node->op() << ":" << node->name();
            RETURN_CACHE(VOID);
        }

        try_run(node);

        RETURN_CACHE(output_proto);
#undef RETURN_CACHE
    }

    std::vector<TensorPrototype>
    infer(std::vector<Node> &nodes, std::unordered_map<Node, TensorPrototype> &cache) {
        std::vector<TensorPrototype> shapes;
        shapes.reserve(nodes.size());
        for (auto &node : nodes) {
            shapes.push_back(infer(node, cache));
        }
        return shapes;
    }

    TensorPrototype infer(Node &node) {
        std::unordered_map<Node, TensorPrototype> cache;
        return infer(node, cache);
    }

    std::vector<TensorPrototype> infer(std::vector<Node> &nodes) {
        std::unordered_map<Node, TensorPrototype> cache;
        return infer(nodes, cache);
    }
}
