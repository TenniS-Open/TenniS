//
// Created by kier on 2018/7/18.
//

#ifndef TENSORSTACK_MODULE_MODULE_H
#define TENSORSTACK_MODULE_MODULE_H

#include <memory>
#include <unordered_map>
#include "core/tensor.h"

#include "graph.h"

namespace ts {
    class Module {
    public:
        using self = Module;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        void load(Graph g);

        void load(Graph g, const std::vector<Node> &outputs);

        void load(Graph g, const std::vector<std::string> &outputs);

        const std::vector<Node> &inputs() const { return m_inputs; }

        const std::vector<Node> &outputs() const { return m_outputs; }

        void clear();

        void sort_inputs(const std::vector<Node> &inputs);

        void sort_inputs(const std::vector<std::string> &input_names);

        void sort_inputs(const std::initializer_list<std::string> &input_names);

        void sort_inputs(const std::initializer_list<const char *> &input_names) {
            sort_inputs(std::vector<std::string>(input_names.begin(), input_names.end()));
        }

        enum SerializationFormat {
            BINARY,
            DESCRIPTION,
        };

        // TODO: finish Load ad Save method
        static Module::shared Load(const std::string &filename, SerializationFormat format = BINARY);

        static void Save(const std::string &filename, Module::shared module, SerializationFormat format = BINARY);

        static std::vector<std::pair<Node, int>> list_reference_nodes(const std::vector<Node> &nodes);

    private:
        /**
         * @param g reference Graph
         * @param outputs output nodes
         * @return input node supporting computing outputs
         */
        static std::vector<Node> graph_walker(Graph g, const std::vector<Node> &outputs);

        std::vector<Node> m_inputs;
        std::vector<Node> m_outputs;
        std::vector<Graph> m_graphs;
    };
}




#endif //TENSORSTACK_MODULE_H
