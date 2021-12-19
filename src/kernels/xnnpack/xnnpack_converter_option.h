//
// Created by sen on 2021/10/27.
//

#ifndef TENNIS_XNNPACK_CONVERTER_OPTION_H
#define TENNIS_XNNPACK_CONVERTER_OPTION_H

#include <backend/name.h>
#include <unordered_set>
#include <unordered_map>
#include <module/io/sstream.h>
#include <global/operator_factory.h>
#include <runtime/workbench.h>
#include "compiler/fence/splitter.h"
#include "module/menu.h"
#include "compiler/option/converter_option.h"
#include "kernels/xnnpack/xnnpack.h"

namespace ts {
    namespace xnn {
        class XnnSplitRule : public SplitRule {
        public:
            bool is_route(const Node &node) const final {
                return m_route.find(node->op()) != m_route.end();
            }

            bool is_support(const Node &node) const final {
                return m_support.find(node->op()) != m_support.end();
            }

            void setSupport(std::unordered_set<std::string> &support_map) {
                m_support = support_map;
            }

            void setRoute(std::unordered_set<std::string> &route_map) {
                m_route = route_map;
            }

        private:
            std::unordered_set<std::string> m_support;
            std::unordered_set<std::string> m_route;
        };

        static std::unordered_set<std::string> xnn_support_op = {
                name::layer::conv2d(),  // 2D convolution
                name::layer::depthwise_conv2d(),
//            name::layer::transpose_conv2d(), // 2D Deconvolution
//                name::layer::pooling2d(), // 2D average pooling and 2D max pooling
                name::layer::conv2d_v2(),

//            name::layer::resize2d(), // only bilinear resize, others using cpu-op
//                name::layer::sample2d(),

                name::layer::add(),
                name::layer::sub(), // Subtract
                name::layer::div(), // Divide
                name::layer::maximum(), // Maximum
//                "minimum", // Minimum not support on cpu
                name::layer::mul(), // Multiply (including broadcasting)
                name::layer::gemm(), // Fully Connected
                "abs",
//                "bankers' rounding", // not support on cpu
                "ceil", // Ceiling
                name::layer::relu(), // Clamp, including relu and relu6

//                "elu", // not support on cpu
                "floor", // Floor
//                "hardswish", // not support on cpu
                "leaky_relu",
                name::layer::sigmoid(),
                name::layer::softmax(),
                name::layer::square(), // Square
                name::layer::prelu(), // PReLU
                name::layer::sqrt(),
                name::layer::relu_max(),
                name::layer::global_pooling2d(),
                "hard_sigmoid",
        };

        static std::unordered_set<std::string> xnn_route_op = {
//                name::layer::batch_norm(),
//                name::layer::batch_scale(),
//                name::layer::inner_prod(),

                name::layer::add_bias(),
                Bubble::Const,
                name::layer::copy(), // Copy, TenniS using shallow copy
                name::layer::cast(),
//                name::layer::concat(),
                // TODO: add element-wise activation
                "softplus",
                "tanh",
                name::layer::to_float(),
                name::layer::flatten(),

        };

        class XnnpackConverter : public ConverterOption {
        public:
            std::shared_ptr<Module> convert(const ComputingDevice &device, const std::vector<Node> &outputs,
                                            const std::vector<Node> &inputs) const override;
        };

        // from src/compiler/fence/splitter.cpp
        // top for output, bottom for input
        class RefCache {
        public:
            bool ref(const Node &top, const Node &bottom) const {
                auto it = m_cache.find(top);
                if (it == m_cache.end()) return false;
                auto bottom_it = it->second.find(bottom);
                if (bottom_it == it->second.end()) return false;
                return true;
            }

            void setup(const std::vector<Node> &outputs, const std::vector<Node> &inputs) {
                for (auto &node : inputs) {
                    m_cache.insert(std::make_pair(node, std::unordered_set<Node>()));
                }
                for (auto &node : outputs) {
                    cache_refs(node);
                }
            }

        private:
            std::unordered_map<Node, std::unordered_set<Node>> m_cache;

            const std::unordered_set<Node> &cache_refs(const Node &node){
                {
                    auto it = m_cache.find(node);
                    if (it != m_cache.end()) return it->second;
                }
                std::unordered_set<Node> refs;
                for (auto &bottom : node.inputs()) {
                    auto &tmp = cache_refs(bottom);
                    refs.insert(tmp.begin(), tmp.end());
                    refs.insert(bottom);
                }
                auto it_succeed = m_cache.insert(std::make_pair(node, std::move(refs)));
                return it_succeed.first->second;
            }
        };

        // 0=not support, greater 0 means color, negative number means no color
        class ColorGraph {
        public:
            void config_color(const Node &node, int n) {
                color(node.ptr(), n);
            }

            void color(const Node &node, int n) {
                {
                    auto pre_n = color(node);
                    if (pre_n >= 0) {
                        m_graphs[pre_n].erase(node);
                    }
                }
                {
                    auto it = m_graphs.find(n);
                    if (it == m_graphs.end()) {
                        m_graphs.insert(std::make_pair(n, std::unordered_set<Node>({node})));
                    } else {
                        it->second.insert(node);
                    }
                }

                color(node.ptr(), n);
            }

            int color(const Node &node) const {
                return color(node.ptr());
            }

            int serial(const Node &node) const {
                return serial(node.ptr());
            }

            const std::unordered_set<Node> &graph(int n) const {
                auto it = m_graphs.find(n);
                if (it == m_graphs.end()) {
                    return m_empty;
                } else {
                    return it->second;
                }
            }

            void clear(const Node &node) {
                auto it = m_color_map.find(node.ptr());
                if (it == m_color_map.end()) return;

                m_graphs[it->second].erase(node);
                m_color_map.erase(it);
            }

            void clear(int i) {
                auto it = m_graphs.find(i);
                if (it == m_graphs.end()) return;
                for (auto &node : it->second) {
                    m_color_map.erase(node.ptr());
                }
                m_graphs.erase(it);
            }

            std::function<bool(const Node &, const Node &)> first_first() const {
                return [&](const Node &a, const Node &b) -> bool {
                    return this->color(a) < this->color(b);
                };
            }

            std::function<bool(const Node &, const Node &)> first_later() const {
                return [&](const Node &a, const Node &b) -> bool {
                    return this->color(a) > this->color(b);
                };
            }

            std::unordered_map<const void *, int> color_map() const {
                return m_color_map;
            }

        private:
            void color(const void *node, int n) {
                m_color_map[node] = n;
                if (m_serial_map.find(node) == m_serial_map.end()) {
                    m_serial_map.insert(std::make_pair(node, m_serial_map.size()));
                }
            }

            int color(const void *node) const {
                auto it = m_color_map.find(node);
                if (it == m_color_map.end()) return -1;
                return it->second;
            }

            int serial(const void *node) const {
                auto it = m_serial_map.find(node);
                if (it == m_serial_map.end()) return -1;
                return it->second;
            }

            std::unordered_map<const void *, int> m_color_map;
            std::unordered_map<const void *, int> m_serial_map;
            std::unordered_map<int, std::unordered_set<Node>> m_graphs;
            std::unordered_set<Node> m_empty;
        };

        class QueueEmptyException : public Exception {};

        class TopFirstQueue {
        public:
            explicit TopFirstQueue(const RefCache *ref)
                    : m_ref(ref) {}

            bool empty() const {
                return m_list.empty();
            }

            void push(const Node &node) {
                m_list.push_back(node);
            }

            void push(const std::vector<Node> &nodes) {
                for (auto &node : nodes) m_list.push_back(node);
            }

            Node pop() {
                if (empty()) throw QueueEmptyException();
                auto it = this->top_it();
                auto node = *it;
                m_list.erase(it);
                return node;
            }

            const Node &top() const {
                if (empty()) throw QueueEmptyException();
                auto it = this->top_it();
                return *it;
            }
        private:
            std::list<Node> m_list;
            const RefCache *m_ref;

            auto top_it() const -> decltype(m_list.begin()) {
                auto &ref = *m_ref;
                auto it = m_list.begin();
                if (it == m_list.end()) return it;
                auto target = it;
                while (true) {
                    ++it;
                    if (it == m_list.end()) break;
                    if (ref.ref(*it, *target)) {
                        target = it;
                    }
                }
                return target;
            }

            auto top_it() -> decltype(m_list.begin()) {
                auto &ref = *m_ref;
                auto it = m_list.begin();
                if (it == m_list.end()) return it;
                auto target = it;
                while (true) {
                    ++it;
                    if (it == m_list.end()) break;
                    if (ref.ref(*it, *target)) {
                        target = it;
                    }
                }
                return target;
            }
        };

        class BottomFirstQueue {
        public:
            explicit BottomFirstQueue(const RefCache *ref)
                    : m_ref(ref) {}

            bool empty() const {
                return m_list.empty();
            }

            void push(const Node &node) {
                m_list.push_back(node);
            }

            void push(const std::vector<Node> &nodes) {
                for (auto &node : nodes) m_list.push_back(node);
            }

            Node pop() {
                if (empty()) throw QueueEmptyException();
                auto it = this->top_it();
                auto node = *it;
                m_list.erase(it);
                return node;
            }

            const Node &top() const {
                if (empty()) throw QueueEmptyException();
                auto it = this->top_it();
                return *it;
            }
        private:
            std::list<Node> m_list;
            const RefCache *m_ref;

            auto top_it() const -> decltype(m_list.begin()) {
                auto &ref = *m_ref;
                auto it = m_list.begin();
                if (it == m_list.end()) return it;
                auto target = it;
                while (true) {
                    ++it;
                    if (it == m_list.end()) break;
                    if (ref.ref(*target, *it)) {
                        target = it;
                    }
                }
                return target;
            }

            auto top_it() -> decltype(m_list.begin()) {
                auto &ref = *m_ref;
                auto it = m_list.begin();
                if (it == m_list.end()) return it;
                auto target = it;
                while (true) {
                    ++it;
                    if (it == m_list.end()) break;
                    if (ref.ref(*target, *it)) {
                        target = it;
                    }
                }
                return target;
            }
        };

        namespace {
            class InputOutput {
            public:
                std::vector<Node> inputs;
                std::vector<Node> outputs;
                bool is_sub_graph = false;
                int color = 0;
            };
        }

        /**
         * get subgraph's input and outputs, update color.
         * @param node
         * @param config
         * @param ref
         * @param rule
         * @param color
         * @return
         */
        static InputOutput explore(const Node &node, const Splitter &config, RefCache &ref,
                                   SplitRule &rule, ColorGraph &color,
                                   int i) {
            InputOutput io;
            io.is_sub_graph = false;

            // bottom up
            std::unordered_set<Node> top_edge; // contain not in graph node at top edge
            std::unordered_set<Node> bottom_edge;

            TopFirstQueue down_finder(&ref);
            BottomFirstQueue up_finder(&ref);

            auto in_this_graph = [&](const Node &n) -> bool {
                return color.color(n) == i;
            };

            auto all_in_this_graph = [&](const std::vector<Node> &ns) -> bool {
                for (auto &n : ns) {
                    if (color.color(n) != i) return false;
                }
                return true;
            };

            auto is_ok_to_add = [&](const Node &n) -> bool {
                if (color.color(n) < 0 && rule.is_route_or_support(n)) {
                    return true;
                }
                return false;
            };

            auto add_to_graph = [&](const Node &n) {
                color.color(n, i);
            };

            auto has_walked = [&](const Node &n) -> bool {
                if (top_edge.find(n) != top_edge.end()) return true;
                if (bottom_edge.find(n) != bottom_edge.end()) return true;
                return color.color(n) == i; // in graph node must be walked
            };

            // bottom up, check outputs
            auto look_up = [&](const Node &n) {
                for (auto &t : n.outputs()) {
                    if (top_edge.find(t) != top_edge.end()) continue;
                    if (in_this_graph(t)) continue;
                    up_finder.push(t);
                }
            };

            // top down, check inputs
            auto look_down = [&](const Node &n) {
                for (auto &t : n.inputs()) {
                    if (bottom_edge.find(t) != bottom_edge.end()) continue;
                    if (in_this_graph(t)) continue;
                    down_finder.push(t);
                }
            };

            auto found_bottom = [&](const Node &n) {
                bottom_edge.insert(n);
            };

            auto found_top = [&](const Node &n) {
                top_edge.insert(n);
            };

            if (!is_ok_to_add(node)) return io;

            add_to_graph(node);
            look_up(node);
            look_down(node);

            auto my_precious = [&](const Node &n) -> bool {
                // if is base supported node
                if (is_ok_to_add(n)) {
                    // check single output
                    if (config.single_output) {
                        if (!all_in_this_graph(n.outputs())) return false;
                    }
                    // check loop case
                    for (auto &t : bottom_edge) {
                        if (ref.ref(t, n)) return false;
                    }
                    for (auto &t : top_edge) {
                        if (ref.ref(n, t)) return false;
                    }
                    return true;
                }
                return false;
            };

            while (true) {
                // top down
                if (down_finder.empty()) break;
                do {
                    auto n = down_finder.pop();
                    if (has_walked(n)) continue;
                    if (my_precious(n)) {
                        add_to_graph(n);
                        look_down(n);
                        look_up(n);
                    } else {
                        found_bottom(n);
                    }
                } while (!down_finder.empty());
                if (config.single_output) break;
                if (up_finder.empty()) break;
                do {
                    auto n = up_finder.pop();
                    if (has_walked(n)) continue;
                    if (my_precious(n)) {
                        add_to_graph(n);
                        look_down(n);
                        look_up(n);
                    } else {
                        found_top(n);
                    }
                } while (!up_finder.empty());
            }

            // get inputs and outputs
            std::unordered_set<Node> graph_tops;
            for (auto &n : top_edge) {
                for (auto &t : n.inputs()) {
                    if (in_this_graph(t)) {
                        graph_tops.insert(t);
                    }
                }
            }
            std::unordered_set<Node> graph_bottoms;
            for (auto &n : bottom_edge) {
                for (auto &t : graph_tops) {
                    if (ref.ref(t, n)) {
                        graph_bottoms.insert(n);
                        break;
                    }
                }
            }

            if (config.single_input) {
                // update graph to match single input
                while (graph_bottoms.size() > 1) {
                    if (color.graph(i).size() <= 1) {
                        color.clear(i);
                        return io;  // not a graph
                    }
                    // find next bottom
                    std::unordered_set<Node> may_remove_set;
                    for (auto &n : graph_bottoms) {
                        for (auto &t : n.outputs()) {
                            if (in_this_graph(t)) {
                                may_remove_set.insert(t);
                            }
                        }
                    }
                    std::vector<Node> may_remove(may_remove_set.begin(), may_remove_set.end());
                    std::sort(may_remove.begin(), may_remove.end(), color.first_later());
                    Node target = may_remove[0];
                    bool side = !ref.ref(node, target);
                    for (size_t j = 1; j < may_remove.size(); ++j) {
                        auto tmp = may_remove[j];
                        if (target == node) {
                            target = tmp;
                            side = !ref.ref(node, target);
                            continue;
                        } else if (tmp == node) {
                            continue;
                        }
                        if (side && ref.ref(node, tmp)) {
                            continue;
                        }
                        if (ref.ref(target, tmp)) {
                            target = tmp;
                            side = !ref.ref(node, target);
                        }
                    }
                    // remove from graph
                    color.clear(target);    // not in this graph
                    // update bottom edge
                    std::unordered_set<Node> remain_edge;
                    may_remove_set.erase(target);
                    for (auto &n : may_remove_set) {
                        auto ns = n.inputs();
                        remain_edge.insert(ns.begin(), ns.end());
                    }
                    std::unordered_set<Node> new_edge;
                    std::set_intersection(std::begin(graph_bottoms), std::end(graph_bottoms),
                                          std::begin(remain_edge), std::end(remain_edge),
                                          std::inserter(new_edge, new_edge.end()));
                    new_edge.insert(target);
                    graph_bottoms = new_edge;
                    if (graph_tops.find(target) != graph_tops.end()) {
                        graph_tops.erase(target);
                        for (auto it = graph_bottoms.begin(); it != graph_bottoms.end(); ) {
                            bool found_ref = false;
                            for (auto &t : graph_tops) {
                                if (ref.ref(t, *it)) {
                                    found_ref = true;
                                    break;
                                }
                            }
                            if (found_ref) {
                                ++it;
                            } else {
                                it = graph_bottoms.erase(it);
                            }
                        }
                    }
                }
            }

            // check route graph, not support all nodes are route graph
            {
                bool has_support = false;
                for (auto &n : color.graph(i)) {
                    if (rule.is_support(n)) {
                        has_support = true;
                        break;
                    }
                }
                if (!has_support) {
                    auto route_graph = color.graph(i);
                    for (auto &n : route_graph) {
                        color.color(n, 0);
                    }

                    io.is_sub_graph = true;
                    io.color = 0;
                    io.inputs = std::vector<Node>(graph_bottoms.begin(), graph_bottoms.end());
                    io.outputs = std::vector<Node>(graph_tops.begin(), graph_tops.end());
                    return io;
                }
            }

            io.color = i;
            io.is_sub_graph = true;
            io.inputs = std::vector<Node>(graph_bottoms.begin(), graph_bottoms.end());
            io.outputs = std::vector<Node>(graph_tops.begin(), graph_tops.end());

            return io;
        }

    }
}

#endif //TENNIS_XNNPACK_CONVERTER_OPTION_H
