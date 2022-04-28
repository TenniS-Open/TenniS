//
// Created by sen on 2021/9/28.
//
#include "compiler/option/converter_option.h"
#include "xnnpack_converter_option.h"

namespace ts {
    namespace xnn {
        static Node clone_node(const Node &node, std::unordered_map<Node, Node> &cloned_nodes) {
            auto it = cloned_nodes.find(node);
            if (it != cloned_nodes.end()) return it->second;
            auto &g = ctx::of<Graph>::ref();
            auto dolly = g.make(node.bubble());
            cloned_nodes.insert(std::make_pair(node, dolly));
            cloned_nodes.insert(std::make_pair(dolly, dolly));
            return dolly;
        }

        static std::vector<Node>
        clone_graph(const std::vector<Node> &nodes, std::unordered_map<Node, Node> &cloned_nodes,
                    std::unordered_map<Node, Node> &linked_nodes) {
            std::vector<Node> dolly_nodes;
            for (auto &node: nodes) {
                {
                    auto it = linked_nodes.find(node);
                    if (it != linked_nodes.end()) {
                        dolly_nodes.emplace_back(it->second);
                        continue;
                    }
                }

                auto dolly_inputs = clone_graph(node.inputs(), cloned_nodes, linked_nodes);
                auto dolly_node = clone_node(node, cloned_nodes);
                Node::Link(dolly_node, dolly_inputs);
                dolly_nodes.emplace_back(dolly_node);
                linked_nodes.insert(std::make_pair(node, dolly_node));
                linked_nodes.insert(std::make_pair(dolly_node, dolly_node));
            }
            return dolly_nodes;
        }

        std::shared_ptr<Module> getSplitModuleBySplitter(const std::vector<Node> &outputs, const std::vector<Node> &inputs,
                                              const Splitter &config, XnnSplitRule &rule) {
            Graph g;
            ctx::bind<Graph> _bind(g);

            ColorGraph color;   // sub graph of route information

            std::vector<Node> bottoms;
            std::vector<Node> tops;
            std::vector<Node> ends;
            std::unordered_map<Node, Node> cloned_nodes;
            std::unordered_map<Node, Node> linked_nodes;
            {

                tops = clone_graph(outputs, cloned_nodes, linked_nodes);
                for (auto &node: inputs) {
                    bottoms.emplace_back(clone_node(node, cloned_nodes));
                }

                for (auto &node: tops) {
                    auto end = g.make("fake::end", node->name() + "_fake_end");
                    Node::Link(end, {node});
                    ends.push_back(end);
                    // color.color(end, 0);
                }
            }

            RefCache ref;
            ref.setup(ends, bottoms);

            TopFirstQueue queue(&ref);
            for (auto &node: tops) queue.push(node);

            std::vector<InputOutput> sub_graph_ios;
            while (!queue.empty()) {
                auto node = queue.pop();
                if (color.color(node) >= 0) continue;
                auto io = explore(node, config, ref, rule, color, int(sub_graph_ios.size() + 1));
                if (!io.is_sub_graph) {
                    color.color(node, 0);
                    for (auto &bottom: node.inputs()) {
                        if (color.color(bottom) >= 0) continue;
                        queue.push(bottom);
                    }
                } else if (io.color == 0) {
                    for (auto &bottom: io.inputs) {
                        if (color.color(bottom) >= 0) continue;
                        queue.push(bottom);
                    }
                } else {
                    sub_graph_ios.push_back(io);
                    for (auto &bottom: io.inputs) {
                        if (color.color(bottom) >= 0) continue;
                        queue.push(bottom);
                    }
                }
            }

            // merge sub graph
            if (!config.single_input && !config.single_output) {
                // TODO: merge graphs
            }
            // deal with max out
            if (config.only_max_graph && sub_graph_ios.size() > 1) {
                size_t max = 0;
                for (size_t i = 1; i < sub_graph_ios.size(); ++i) {
                    if (color.graph(sub_graph_ios[i].color).size() >
                        color.graph(sub_graph_ios[max].color).size()) {
                        max = i;
                    }
                }
                sub_graph_ios = {sub_graph_ios[max]};
            }

            // remove small graph
            {
                for (auto it = sub_graph_ios.begin(); it != sub_graph_ios.end();) {
                    if (color.graph(it->color).size() < config.min_graph_size) {
                        // set node color to 0, which unsatisfying config.min_graph_size
                        for (const auto &colored_node: color.graph(it->color)) {
                            if (Bubble::IsEndPoint(colored_node->op())) continue;
                            color.config_color(colored_node, 0);
                        }
                        it = sub_graph_ios.erase(it);
                    } else {
                        ++it;
                    }
                }
            }

            auto collect_node_inputs = [&](const Node& node, const Node& node_in, Node& node_inserted, std::vector<Node>& linking) -> void {
                for (auto &out_node_in: node.inputs()) {
                    if (out_node_in->name() == node_in->name()) {
                        linking.emplace_back(node_inserted);
                        continue;
                    }
                    linking.emplace_back(out_node_in);
                }
            };

            for (auto &node_iter: cloned_nodes) {
                Node node = node_iter.first;

                if (Bubble::IsEndPoint(node->op())) continue;

                std::string xnn_op_prefix = "xnn::";
                if (color.color(node) > 0) {
                    if (xnn_support_op.find(node->op()) != xnn_support_op.end()) {
                        node.bubble().op(xnn_op_prefix + node->op());
                    }

                    for (auto &node_in: node.inputs()) {
                        if (color.color(node_in) == 0) {
                            Node node_inserted = g.make(name::layer::transpose(), node->name() + "_nchw2nhwc");
                            node_inserted->set(name::permute, tensor::from({0, 2, 3, 1}));
                            for (auto &node_out: node_in.outputs()) {
                                // confirm the colored branch
                                if (color.color(node_out) > 0) {
                                    std::vector<Node> linking;
                                    collect_node_inputs(node_out, node_in, node_inserted, linking);
                                    Node::Link(node_out, linking);
                                }
                            }
                            Node::Link(node_inserted, {node_in});
                        }
                    }

                    // if dims of operator output less 4, do not transpose
                    if (node->op() == xnn_op_prefix + name::layer::gemm()) {
                        continue;
                    }

                    Node node_inserted = g.make(name::layer::transpose(), node->name() + "_nhwc2nchw");
                    node_inserted->set(name::permute, tensor::from({0, 3, 1, 2}));

                    for (auto &node_out : node.outputs()) {
                        // some node have colored to -1
                        if (color.color(node_out) <= 0) {
                            std::vector<Node> linking;
                            Node::Link(node_inserted, {node});
                            collect_node_inputs(node_out, node, node_inserted, linking);
                            Node::Link(node_out, linking);
                        }
                    }
                }
            }

            auto main_inputs = bottoms;
            std::vector<Node> main_outputs;
            for (auto &n : ends) {
                main_outputs.emplace_back(n.inputs()[0]);
                Node::Link(n, {});
            }

            auto xnnpack_module = std::make_shared<Module>();
            xnnpack_module->load(g, main_outputs);
            xnnpack_module->sort_inputs(main_inputs);
            return xnnpack_module;
        }

        std::shared_ptr<Module>
        XnnpackConverter::convert(const ComputingDevice &device, const std::vector<Node> &outputs,
                                  const std::vector<Node> &inputs) const {
            XnnSplitRule rule;
            rule.setSupport(xnn_support_op);
            rule.setRoute(xnn_route_op);
            Splitter splitter(&rule);
            // TODO: check the size
            splitter.min_graph_size = 5;
            splitter.submodule = "xnnpack";
            splitter.single_input = false;
            splitter.single_output = false;
//            splitter.only_max_graph = true;

            auto module = getSplitModuleBySplitter(outputs, inputs, splitter, rule);
            auto translated_module = Module::Translate(module, CPU, "--xnnpack");
            // which xnn backend do not support, use pack optimization
            translated_module = Module::Translate(translated_module, CPU, "--pack");
            return translated_module;
        }
    }
}

using namespace ts;
using namespace xnn;
#ifdef TS_USE_XNNPACK
TS_REGISTER_CONVERTER_OPTION(XNNPACK, XnnpackConverter)
#endif