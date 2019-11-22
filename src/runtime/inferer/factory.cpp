//
// Created by kier on 2019/11/22.
//

#include <core/tensor_builder.h>
#include "global/shape_inferer_factory.h"
#include "utils/static.h"
#include "backend/name.h"

namespace ts {
    namespace infer {
        static Tensor get_value(const Node &node) {
            if (node->op() == Bubble::Const) {
                return node->get("value");
            }
            if (node->has("#value")) {
                return node->get("#value");
            }
            return Tensor();
        }

        static TensorPrototype _param(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (!node->has("#shape")) {
                throw Exception(node->op() + ":" + node->name() + " must set #shape");
                // return TensorPrototype();
            }
            auto dtype = FLOAT32;
            if (node->has("#dtype")) {
                dtype = DTYPE(tensor::to_int(node->get("#dtype")));
            }
            auto shape = tensor::array::to_int(node->get("#shape"));
            return TensorPrototype(dtype, shape);
        }

        static TensorPrototype _const(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto value = node->get("value");
            return value.proto();
        }

        static TensorPrototype _resize2d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto x = inputs[0];
            auto size_value = get_value(node.input(1));
            if (size_value.empty()) return VOID;
            auto size = tensor::array::to_int(size_value);
            auto y_shape = x.sizes();
            if (size.size() != y_shape.size()) return TensorPrototype();
            for (size_t i = 0; i < size.size(); ++i) {
                if (size[i] > 0) y_shape[i] = size[i];
            }
            return TensorPrototype(x.dtype(), y_shape);
        }
    }

    TS_STATIC_ACTION(ShapeInferer::Register, "<param>", infer::_param)
    TS_STATIC_ACTION(ShapeInferer::Register, "<const>", infer::_const)
    TS_STATIC_ACTION(ShapeInferer::Register, "_resize2d", infer::_resize2d)
}
