//
// Created by seeta on 2018/7/18.
//

#ifndef TENSORSTACK_MODULE_MODULE_H
#define TENSORSTACK_MODULE_MODULE_H

#include <memory>

#include "graph.h"

namespace ts {
    class OP {
    public:
        explicit OP(
                const std::string &op,
                const std::string &name = "")
                : op(op), name(name) {}

        std::string op;
        std::string name;

        static const char *const Parameter; // Mean must input in graph
        static const char *const Const;     // Mean never change in graph, including weights, saving in static memory
        static const char *const Variable;  // Mean may change in graph, saving some state, do no use in this version

        static bool IsEndPoint(const std::string &op);
    };

    inline std::ostream &operator<<(std::ostream &out, const OP &op) {
        return out << op.op << ", " << op.name;
    }

    class Module {
    public:
        using self = Module;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        void load(Graph g);
        void load(Graph g, const std::vector<Node> &outputs);
        void load(Graph g, const std::vector<std::string> &outputs);

        const std::vector<Node> &inputs() const { return m_inputs; }

        const std::vector<Node> &outputs() const { return m_outputs; }

    private:
        std::vector<Node> m_inputs;
        std::vector<Node> m_outputs;
        std::vector<Graph> m_graphs;
    };
}



#endif //TENSORSTACK_MODULE_H
