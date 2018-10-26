//
// Created by seeta on 2018/7/18.
//

#include <set>
#include <queue>
#include "module/module.h"

namespace ts {
    const char *const OP::IN = "<in>";

    void Module::load(Graph g) {
        // calculate inputs and outputs
        auto nodes = g.nodes();
        std::set<Node> nodes_set;
        std::vector<Node> outputs;
        for (auto &node : nodes) {
            nodes_set.insert(node);
            if (node.outputs().empty()) {
                outputs.push_back(node);
            }
        }
        // valid graph, get inputs
        std::vector<Node> inputs;
        std::queue<Node> walker;
        std::set<Node> walked;
        for (auto &node : outputs) walker.push(node);
        while (!walker.empty()) {
            auto node = walker.front();
            walker.pop();
            if (walked.find(node) != walked.end()) continue;
            walked.insert(node);
            if (nodes_set.find(node) == nodes_set.end()) {
                throw ts::Exception("Found unlinked node in graph");
            }
            auto &op = node.ref<OP>();
            if (op.op == OP::IN) {
                inputs.push_back(node);
                continue;
            }
            if (node.inputs().empty()) {
                throw ts::Exception("Found not computable node");
            }
            for (auto &input : node.inputs()) {
                walker.push(input);
            }
        }
        // remove duplicate inputs
        std::set<Node> input_nodes_set;
        for (auto &input : inputs) input_nodes_set.insert(input);
        m_inputs.insert(m_inputs.end(), input_nodes_set.begin(), input_nodes_set.end());
        m_outputs.insert(m_outputs.end(), outputs.begin(), outputs.end());
        m_graphs.push_back(g);
    }
}