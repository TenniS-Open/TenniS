//
// Created by kier on 2018/10/31.
//

#include "module/menu.h"

namespace ts {
    namespace bubble {
        Node param(const std::string &name) {
            auto &g = ctx::ref<Graph>();
            return g.make<Bubble>(Bubble::Parameter, name);
        }

        Node op(const std::string &name, const std::string &op_name, const std::vector<Node> &inputs) {
            auto &g = ctx::ref<Graph>();
            Node result = g.make<Bubble>(op_name, name);
            Node::Link(result, inputs);
            return result;
        }

        Node data(const std::string &name, const Tensor &value) {
            auto &g = ctx::ref<Graph>();
            Node result = g.make<Bubble>(Bubble::Const, name);
            result.ref<Bubble>().set("value", value);
            return result;
        }
    }
}