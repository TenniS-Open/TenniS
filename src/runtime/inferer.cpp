//
// Created by kier on 2019/11/20.
//

#include <core/tensor_builder.h>
#include "runtime/inferer.h"
#include "global/shape_inferer_factory.h"

#include "runtime/stack.h"
#include "global/operator_factory.h"

#include "global/hard_allocator.h"

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

    void *FakeMemoryAllocator(int, size_t, void *, size_t) {
        return nullptr;
    }

    TS_STATIC_ACTION(HardAllocator::Register, "__fake__", FakeMemoryAllocator)

    static bool valid_shape(const Shape &shape) {
        for (auto &dim : shape) {
            if (dim < 0) return false;
        }
        return true;
    }

    void infer_value(Node &node, const std::vector<bool> &ignore) {
        if (Bubble::IsEndPoint(node->op())) {
            return;
        }

        std::vector<Tensor> inputs(node.inputs().size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto input = node.input(i);
            auto this_input = get_value(input);
            if (!this_input.empty()) {
                inputs[i] = this_input;
                continue;
            }
            if (i >= ignore.size()) return;
            if (!ignore[i]) return;
            if (!input->has("#shape")) return;
            auto fake_shape = input->get_int_list("#shape");
            for (auto &dim : fake_shape) {
                if (dim < 0) dim = 1;
            }
            inputs[i] = Tensor(MemoryDevice("__fake__"), FLOAT32, fake_shape);
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

        auto before_node_value = get_value(node);

        auto output_proto = shape_infer(node, input_proto);
        if (output_proto.dtype() == VOID) {
            TS_LOG_ERROR << "Failed to infer " << node->op() << ":" << node->name();
            RETURN_CACHE(VOID);
        }

        auto after_node_value = get_value(node);

        if (after_node_value.empty() || after_node_value.data() == before_node_value.data()) {
            // if node's #value not set or not updated, then do infer
            infer_value(node);
        }

        if (node->op() != Bubble::Const)
            TS_LOG_INFO << node->op() << ":" << node->name() << " => " << output_proto;

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

    void infer_value(Node &node) {
        return infer_value(node, {});
    }

    std::vector<Tensor::Prototype> infer_shape(const Node &node, const std::vector<bool> &ignore) {
        if (node->op() == Bubble::Const) {
            return {node->get("value").proto()};
        }

        if (node->op() == Bubble::Parameter) {
            if (!node->has("#shape")) return {};
            auto dtype = FLOAT32;
            if (node->has("#dtype")) {
                dtype = DTYPE(tensor::to_int(node->get("#dtype")));
            }
            auto shape = tensor::array::to_int(node->get("#shape"));
            return {Tensor::Prototype(dtype, shape)};
        }

        std::vector<Tensor> inputs(node.inputs().size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto input = node.input(i);
            auto this_input = get_value(input);
            if (!this_input.empty()) {
                inputs[i] = this_input;
                continue;
            }
            if (i >= ignore.size()) return {};
            if (!ignore[i]) return {};
            if (!input->has("#shape")) return {};
            auto fake_shape = input->get_int_list("#shape");
            if (!valid_shape(fake_shape)) return {};
            inputs[i] = Tensor(MemoryDevice("__fake__"), FLOAT32, fake_shape);
        }
        MemoryDevice memory_device(CPU);
        Stack stack(memory_device);

        auto op = OperatorCreator::CreateNoException(memory_device.type(), node->op());
        if (op == nullptr) return {};

        std::vector<Tensor::Prototype> output;

        try {
            for (const auto &it : node->params()) {
                op->set(it.first, it.second);
            }
            op->init();
            for (auto &t : inputs) {
                stack.push(t);
            }
            op->infer(stack, output);
        } catch (...) {
            return {};
        }

        return output;
    }

    std::vector<Tensor::Prototype> infer_shape(const Node &node) {
        return infer_shape(node, {});
    }
}
