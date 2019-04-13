//
// Created by kier on 2019/3/27.
//

#include "frontend/zoo.h"

#include "module/menu.h"
#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace symbol {
        Node pad(const std::string &name, const Node &x, const Node &padding, float padding_value) {
            Node node = bubble::op(name, name::layer::pad(), {x, padding});
            node->set(name::padding_value, tensor::from<float>(padding_value));
            return node;
        }

        Node resize(const std::string &name, const Node &x, const Node &size, desc::ResizeType type) {
            Node node = bubble::bubble(desc::resize(type), name);
            Node::Link(node, {x, size});
            return node;
        }
    }
}
