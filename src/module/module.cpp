//
// Created by kier on 2018/7/18.
//

#include <unordered_set>
#include <queue>
#include <cassert>

#include "module/module.h"
#include "utils/box.h"
#include "core/tensor_builder.h"

namespace ts {
    const char *const Bubble::Parameter = "<param>";
    const char *const Bubble::Const = "<const>";
    const char *const Bubble::Variable = "<var>";
    static const std::unordered_set<std::string> EndPoints = {Bubble::Parameter, Bubble::Const, Bubble::Variable};

    bool Bubble::IsEndPoint(const std::string &op) {
        return EndPoints.find(op) != EndPoints.end();
    }

    Bubble::Bubble(const std::string &op)
            : m_op(op) {
        update_retention_params();
    }

    Bubble::Bubble(const std::string &op, const std::string &name)
            : m_op(op)
            , m_name(name) {
        update_retention_params();
    }

    Bubble::Bubble(const std::string &op, size_t output_count)
            : m_op(op)
            , m_output_count(output_count) {
        update_retention_params();
    }

    Bubble::Bubble(const std::string &op, const std::string &name, size_t output_count)
            : m_op(op)
            , m_name(name)
            , m_output_count(output_count) {
        update_retention_params();
    }

    bool Bubble::has(const std::string &param) const {
        return this->m_params.find(param) != this->m_params.end();
    }

    void Bubble::set(const std::string &param, const Tensor &value) {
        this->m_params.insert(std::make_pair(param, value));
    }

    Tensor &Bubble::get(const std::string &param) {
        auto param_it = m_params.find(param);
        if (param_it == m_params.end()) {
            throw Exception(
                    std::string("Unidentified param \"") + param + "\", did you mean \"" + fuzzy_param_name(param) +
                    "\"");
        }
        return param_it->second;
    }

    const Tensor &Bubble::get(const std::string &param) const {
        return const_cast<self *>(this)->get(param);
    }

    void Bubble::clear(const std::string &param) {
        auto param_it = m_params.find(param);
        if (param_it == m_params.end()) {
            throw Exception(
                    std::string("Unidentified param \"") + param + "\", did you mean \"" + fuzzy_param_name(param) +
                    "\"");
        }
        this->m_params.erase(param_it);
    }

    void Bubble::clear_params() {
        std::vector<std::pair<std::string, Tensor>> retention_params;
        for (auto &param_tenosr_pair : m_params) {
            auto &param = param_tenosr_pair.first;
            bool is_retention_param = !param.empty() && param[0] == retention_param_sign;
            if (is_retention_param) {
                retention_params.emplace_back(param_tenosr_pair);
            }
        }
        m_params.clear();
        m_params.insert(retention_params.begin(), retention_params.end());
    }

    std::string Bubble::fuzzy_param_name(const std::string &name) {
        if (m_params.empty()) return "";
        int min_edit_distance = INT_MAX;
        std::string closest_name;
        for (auto &param_tensor_pair : m_params) {
            auto &target_name = param_tensor_pair.first;
            int dist = edit_distance(name, target_name);
            if (dist < min_edit_distance) {
                closest_name = target_name;
                min_edit_distance = dist;
            }
        }
        return closest_name;
    }

    void Bubble::update_retention_params() {
        assert(retention_param_sign == '#');
        set("#op", tensor::from(m_op));
        set("#name", tensor::from(m_name));
        set("#output_count", tensor::from(m_output_count));
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
            if (op.op() == Bubble::Const) {
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