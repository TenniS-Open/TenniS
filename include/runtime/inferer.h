//
// Created by kier on 2019/11/20.
//

#ifndef TENSORSTACK_RUNTIME_INFERER_H
#define TENSORSTACK_RUNTIME_INFERER_H

#include "module/graph.h"

namespace ts {
    TS_DEBUG_API TensorPrototype infer(Node &node, std::unordered_map<Node, TensorPrototype> &cache);

    TS_DEBUG_API std::vector<TensorPrototype> infer(std::vector<Node> &nodes, std::unordered_map<Node, TensorPrototype> &cache);

    TS_DEBUG_API TensorPrototype infer(Node &node);

    TS_DEBUG_API std::vector<TensorPrototype> infer(std::vector<Node> &nodes);
}

#endif // TENSORSTACK_RUNTIME_INFERER_H