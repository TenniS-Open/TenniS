//
// Created by kier on 2018/10/31.
//

#ifndef TENSORSTACK_MODULE_MENU_H
#define TENSORSTACK_MODULE_MENU_H

#include "module.h"
#include "utils/ctxmgr.h"

namespace ts {
    namespace bubble {
        /**
         * get Parameter node
         * @param name Node name
         * @return new Node belonging to context-Graph
         * @note Must call `ts::ctx::bind<Graph>` to bind context firstly
         */
        Node param(const std::string &name);

        /**
         * get Parameter node
         * @param name Node name
         * @param op_name Operator name
         * @param inputs Input nodes
         * @return new Node belonging to context-Graph
         * @note Must call `ts::ctx::bind<Graph>` to bind context firstly
         */
        Node op(const std::string &name, const std::string &op_name, const std::vector<Node> &inputs);

        /**
         * get Parameter node
         * @param name Node name
         * @param value the data value
         * @return new Node belonging to context-Graph
         * @note Must call `ts::ctx::bind<Graph>` to bind context firstly
         */
        Node data(const std::string &name, const Tensor &value);
    }

    /**
     * write nodes to file, ref as nodes
     * @param stream writer
     * @param nodes nodes ready to save to stream
     * @param base node index base in graph
     * @return writen size
     * @note do not parse param `base` in this version
     */
    size_t serialize_nodes(StreamWriter &stream, const std::vector<Node> &nodes, size_t base = 0);

    /**
     * you need call ctx::bind<Graph> first
     * @param stream reader
     * @return read size
     */
    size_t externalize_nodes(StreamReader &stream);

    /**
     * @param stream writer
     * @param graph ready write graph
     * @return writen size
     */
    size_t serialize_graph(StreamWriter &stream, const Graph &graph);

    /**
     * @param stream reader
     * @param graph ready read graph
     * @return read size
     */
    size_t externalize_graph(StreamReader &stream, Graph &graph);
}


#endif //TENSORSTACK_MODULE_MENU_H
