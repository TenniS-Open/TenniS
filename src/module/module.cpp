//
// Created by seeta on 2018/7/18.
//

#include <unordered_set>
#include <queue>
#include "module/module.h"
#include <unordered_set>
#include <module/module.h>

namespace ts {
    const char *const Bubble::Parameter = "<param>";
    const char *const Bubble::Const = "<const>";
    const char *const Bubble::Variable = "<var>";
    static const std::unordered_set<std::string> EndPoints = {Bubble::Parameter, Bubble::Const, Bubble::Variable};

    bool Bubble::IsEndPoint(const std::string &op) {
        return EndPoints.find(op) != EndPoints.end();
    }



    void Bubble::set(const std::string &param, const Tensor &value) {
        this->m_params.insert(std::make_pair(param, value));
    }

    Tensor &Bubble::get(const std::string &param) {
        return this->m_params.at(param);
    }

    const Tensor &Bubble::get(const std::string &param) const {
        return this->m_params.at(param);
    }

    void Bubble::clear(const std::string &param) {
        this->m_params.erase(param);
    }

    void Bubble::clear_params() {
        this->m_params.clear();
    }

    void Module::load(Graph g) {
        // calculate inputs and outputs
        auto nodes = g.nodes();
        std::unordered_set<Node> nodes_set;
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
        std::unordered_set<Node> walked;
        for (auto &node : outputs) walker.push(node);
        while (!walker.empty()) {
            auto node = walker.front();
            walker.pop();
            if (walked.find(node) != walked.end()) continue;
            walked.insert(node);
            if (nodes_set.find(node) == nodes_set.end()) {
                throw ts::Exception("Found unlinked node in graph");
            }
            auto &op = node.ref<Bubble>();
            if (op.op() == Bubble::Parameter) {
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
        std::unordered_set<Node> input_nodes_set;
        for (auto &input : inputs) input_nodes_set.insert(input);
        m_inputs.insert(m_inputs.end(), input_nodes_set.begin(), input_nodes_set.end());
        m_outputs.insert(m_outputs.end(), outputs.begin(), outputs.end());
        m_graphs.push_back(g);
    }
}