//
// Created by Seeta on 2021/3/29.
//

#ifndef TENNIS_SPLITTER_H
#define TENNIS_SPLITTER_H

#include "../../module/graph.h"
#include "../../module/module.h"

#include <unordered_set>

namespace ts {
    class Splitter;

    class TS_DEBUG_API SubGraph {
    public:
        const std::vector<Node> &inputs() const {
            return m_inputs;
        }

        const std::vector<Node> &outputs() const {
            return m_outputs;
        }

        const Graph &graph() const {
            return m_graph;
        }

    private:
        friend class Splitter;

        Graph m_graph;
        std::vector<Node> m_inputs;
        std::vector<Node> m_outputs;
    };

    class TS_DEBUG_API MainGraph {
    public:
        const SubGraph &sub_graph(size_t i) const {
            return m_sub_graphs[i];
        }

        size_t sub_count() const {
            return m_sub_graphs.size();
        }

        const Node &sub_node(size_t i) const {
            return m_sub_nodes[i];
        }

        Node &sub_node(size_t i) {
            return m_sub_nodes[i];
        }

        const std::vector<Node> &inputs() const {
            return m_module->inputs();
        }

        const std::vector<Node> &outputs() const {
            return m_module->outputs();
        }

        std::shared_ptr<Module> module() const {
            return m_module;
        }

        const std::vector<SubGraph> &graphs() const {
            return m_sub_graphs;
        }

    private:
        friend class Splitter;

        std::shared_ptr<Module> m_module;
        std::vector<Node> m_sub_nodes;
        std::vector<SubGraph> m_sub_graphs;
    };

    class TS_DEBUG_API SplitRule {
    public:
        using self = SplitRule;

        ~SplitRule() = default;

        virtual bool is_route(const Node &node) const = 0;

        virtual bool is_support(const Node &node) const = 0;

        bool is_route_or_support(const Node &node) const {
            return is_route(node) || is_support(node);
        }
    };

    class TS_DEBUG_API Splitter {
    public:
        using self = Splitter;

        Splitter(const Splitter &) = delete;

        Splitter &operator=(const Splitter &) = delete;

        bool only_max_graph = false;
        bool single_input = false;
        bool single_output = false;
        int min_graph_size = 0;
        std::string submodule = "submodule";

        explicit Splitter(SplitRule *rule);

        MainGraph split(const std::vector<Node> &outputs, const std::vector<Node> &inputs);

    private:
        SplitRule *m_rule;
    };
}


#endif //TENNIS_SPLITTER_H
