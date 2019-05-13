//
// Created by kier on 2019/3/27.
//

#include "frontend/symbol.h"

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

        Node resize2d(const std::string &name, const Node &x, const Node &size, desc::ResizeType type) {
            Node node = bubble::bubble(desc::resize2d(type), name);
            Node::Link(node, {x, size});
            return node;
        }

        Node add(const std::string &name, const Node &lhs, const Node &rhs) {
            Node node = bubble::bubble(desc::add(), name);
            Node::Link(node, {lhs, rhs});
            return node;
        }

        Node sub(const std::string &name, const Node &lhs, const Node &rhs) {
            Node node = bubble::bubble(desc::sub(), name);
            Node::Link(node, {lhs, rhs});
            return node;
        }

        Node mul(const std::string &name, const Node &lhs, const Node &rhs) {
            Node node = bubble::bubble(desc::mul(), name);
            Node::Link(node, {lhs, rhs});
            return node;
        }

        Node div(const std::string &name, const Node &lhs, const Node &rhs) {
            Node node = bubble::bubble(desc::div(), name);
            Node::Link(node, {lhs, rhs});
            return node;
        }

        Node transpose(const std::string &name, const Node &x, const std::vector<int32_t> &permute) {
            Node node = bubble::bubble(desc::transpose(permute), name);
            Node::Link(node, {x});
            return node;
        }
    }
}
