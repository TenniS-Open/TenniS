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
    class Bubble {
    public:
        using self = Bubble;

        template<typename K, typename V>
        using map = std::unordered_map<K, V>;
        using param_dict = map<std::string, Tensor>;

        const static char retention_param_sign = '#';

        explicit Bubble(
                const std::string &op);

        explicit Bubble(
                const std::string &op,
                const std::string &name);

        explicit Bubble(
                const std::string &op,
                int output_count);

        explicit Bubble(
                const std::string &op,
                const std::string &name,
                int output_count);

        size_t output_count() const { return m_output_count; }

        const std::string &op() const { return m_op; }

        const std::string &name() const { return m_name; }

        const param_dict &params() const { return m_params; }

        bool has(const std::string &param) const;

        void set(const std::string &param, const Tensor &value);

        Tensor &get(const std::string &param);

        const Tensor &get(const std::string &param) const;

        void clear(const std::string &param);

        void clear_params();

    private:
        void update_retention_params();

        /**
         * Operator name
         */
        std::string m_op;
        /**
         * Bubble name
         */
        std::string m_name;
        /**
         * Saving output size
         */
        int m_output_count = 1;
        /// TODO: Since output count must greater than 1, try supporting 0 output

        /**
         * Parameters
         */
        param_dict m_params;

        std::string fuzzy_param_name(const std::string &name);

    public:
        static const char *const Parameter; // Mean must input in graph
        static const char *const Const;     // Mean never change in graph, including weights, saving in static memory
        static const char *const Variable;  // Mean may change in graph, saving some state, do no use in this version

        static bool IsEndPoint(const std::string &op);
    };

    inline std::ostream &operator<<(std::ostream &out, const Bubble &op) {
        return out << op.op() << ", " << op.name() << ", " << op.output_count();
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

        void clear();

        void sort_inputs(const std::vector<Node> &inputs);

        void sort_inputs(const std::vector<std::string> &input_names);

        void sort_inputs(const std::initializer_list<std::string> &input_names);

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
