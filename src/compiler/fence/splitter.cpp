//
// Created by Seeta on 2021/3/29.
//

#include "compiler/fence/splitter.h"

#include <algorithm>
#include <list>
#include <algorithm>
#include <random>

#include "module/menu.h"

namespace ts {
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

    static Node clone_node(const Node &node, std::unordered_map<Node, Node> &cloned_nodes) {
        auto it = cloned_nodes.find(node);
        if (it != cloned_nodes.end()) return it->second;
        auto &g = ctx::of<Graph>::ref();
        auto dolly = g.make(node.bubble());
        cloned_nodes.insert(std::make_pair(node, dolly));
        cloned_nodes.insert(std::make_pair(dolly, dolly));
        return dolly;
    }

    static std::vector<Node> clone_graph(const std::vector<Node> &nodes, std::unordered_map<Node, Node> &cloned_nodes, std::unordered_map<Node, Node> &linked_nodes) {
        std::vector<Node> dolly_nodes;
        for (auto &node : nodes) {
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
                return io;
            }
        }

        io.color = i;
        io.is_sub_graph = true;
        io.inputs = std::vector<Node>(graph_bottoms.begin(), graph_bottoms.end());
        io.outputs = std::vector<Node>(graph_tops.begin(), graph_tops.end());

        return io;
    }

    MainGraph Splitter::split(const std::vector<Node> &outputs, const std::vector<Node> &inputs) {
        Graph g;
        ctx::bind<Graph> _bind(g);

        ColorGraph color;   // sub graph of route information
        size_t node_count = 0;

        std::vector<Node> bottoms;
        std::vector<Node> tops;
        std::vector<Node> ends;
        {
            std::unordered_map<Node, Node> cloned_nodes;
            std::unordered_map<Node, Node> linked_nodes;

            tops = clone_graph(outputs, cloned_nodes, linked_nodes);
            for (auto &node : inputs) {
                bottoms.emplace_back(clone_node(node, cloned_nodes));
            }
            node_count = cloned_nodes.size();

            for (auto &node : tops) {
                auto end = g.make("fake::end", node->name() + "_fake_end");
                Node::Link(end, {node});
                ends.push_back(end);
                // color.color(end, 0);
            }
        }

        RefCache ref;
        ref.setup(ends, bottoms);

        TopFirstQueue queue(&ref);
        for (auto &node : tops) queue.push(node);

        std::vector<InputOutput> sub_graph_ios;
        while (!queue.empty()) {
            auto node = queue.pop();
            if (color.color(node) >= 0) continue;
            auto io = explore(node, *this, ref, *this->m_rule, color, int(sub_graph_ios.size() + 1));
            if (!io.is_sub_graph) {
                color.color(node, 0);
                for (auto &bottom : node.inputs()) {
                    if (color.color(bottom) >= 0) continue;
                    queue.push(bottom);
                }
            } else if (io.color == 0) {
                for (auto &bottom : io.inputs) {
                    if (color.color(bottom) >= 0) continue;
                    queue.push(bottom);
                }
            } else {
                sub_graph_ios.push_back(io);
                for (auto &bottom : io.inputs) {
                    if (color.color(bottom) >= 0) continue;
                    queue.push(bottom);
                }
            }
        }

        // merge sub graph
        if (!single_input && !single_output) {
            // TODO: merge graphs
        }
        // deal with max out
        if (only_max_graph && sub_graph_ios.size() > 1) {
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
            for (auto it = sub_graph_ios.begin(); it != sub_graph_ios.end(); ) {
                if (color.graph(it->color).size() < min_graph_size) {
                    it = sub_graph_ios.erase(it);
                } else {
                    ++it;
                }
            }
        }

        // sort input or output nodes
        for (auto &io : sub_graph_ios) {
            // sort input
            {
                std::sort(io.inputs.begin(), io.inputs.end(), color.first_first());
                std::vector<std::pair<int64_t, Node>> prior_inputs;
                for (size_t j = 0; j < io.inputs.size(); ++j) {
                    auto &n = io.inputs[j];
                    auto prior = node_count - color.serial(n);
                    for (size_t k = 0; k < bottoms.size(); ++k) {
                        auto &t = bottoms[k];
                        if (n == t) {
                            prior += (bottoms.size() - k) * node_count;
                            break;
                        }
                        if (ref.ref(n, t)) {
                            prior += (bottoms.size() - k) * node_count;
                            break;
                        }
                    }
                    prior_inputs.emplace_back(std::make_pair(prior, n));
                }
                std::sort(prior_inputs.begin(), prior_inputs.end(),
                          [](const std::pair<int64_t, Node> &a, const std::pair<int64_t, Node> &b) {
                    return a.first > b.first;
                });
                io.inputs.clear();
                for (auto &n : prior_inputs) io.inputs.emplace_back(n.second);
            }
            // sort output
            {
                std::sort(io.outputs.begin(), io.outputs.end(), color.first_first());
                std::vector<std::pair<int64_t, Node>> prior_outputs;
                for (size_t j = 0; j < io.outputs.size(); ++j) {
                    auto &n = io.outputs[j];
                    auto prior = node_count - color.serial(n);
                    for (size_t k = 0; k < tops.size(); ++k) {
                        auto &t = tops[k];
                        if (n == t) {
                            prior += (tops.size() - k) * node_count;
                            break;
                        }
                        if (ref.ref(t, n)) {
                            prior += (tops.size() - k) * node_count;
                            break;
                        }
                    }
                    prior_outputs.emplace_back(std::make_pair(prior, n));
                }
                std::sort(prior_outputs.begin(), prior_outputs.end(),
                          [](const std::pair<int64_t, Node> &a, const std::pair<int64_t, Node> &b) {
                              return a.first > b.first;
                          });
                io.outputs.clear();
                for (auto &n : prior_outputs) io.outputs.emplace_back(n.second);
            }
        }

        // build return values
        MainGraph main;

        for (auto &io : sub_graph_ios) {
            std::vector<Node> sub_outputs;  // new node contain all fused module node
            std::vector<Node> sub_inputs;   // new node contain all input params

            std::unordered_map<Node, Node> map_origin2sub;
            std::vector<Node> sub_node;

            // new output
            if (io.outputs.size() == 1) {
                auto sub = bubble::op(io.outputs[0]->name(), submodule, {});
                sub_outputs.push_back(sub);
                map_origin2sub.insert(std::make_pair(io.outputs[0], sub));
                Node::Link(sub, io.inputs);
                sub_node.emplace_back(sub);
            } else {
                std::ostringstream oss;
                for (size_t j = 0; j < io.outputs.size(); ++j) {
                    if (j) oss << "&";
                    oss << io.outputs[j]->name();
                }
                auto sub = bubble::op(oss.str(), submodule, {});
                for (size_t j = 0; j < io.outputs.size(); ++j) {
                    auto &n = io.outputs[j];
                    auto out = bubble::op(n->name(), "_field", {sub});
                    out->set("offset", tensor::build(INT32, int32_t(j)));
                    sub_outputs.push_back(out);
                    map_origin2sub.insert(std::make_pair(n, out));
                }
                Node::Link(sub, io.inputs);
                sub_node.emplace_back(sub);
            }
            // new input
            for (auto &n : io.inputs) {
                auto in = bubble::param(n->name());
                if (n->has("#dtype")) {
                    in->dtype(n->dtype());
                }
                if (n->has("#shape")) {
                    in->shape(n->shape());
                }
                sub_inputs.emplace_back(in);
                map_origin2sub.insert(std::make_pair(n, in));
            }
            for (auto &n : io.inputs) {
                auto n_outputs = n.outputs();
                for (auto &t : n_outputs) {
                    if (color.color(t) != io.color) continue;
                    std::vector<Node> t_links;
                    for (auto &k : t.inputs()) {
                        auto it = map_origin2sub.find(k);
                        if (it == map_origin2sub.end()) {
                            t_links.emplace_back(k);
                        } else {
                            t_links.emplace_back(it->second);
                        }
                    }
                    Node::Link(t, t_links);
                }
            }
            // replace output
            for (auto &n : io.outputs) {
                auto n_outputs = n.outputs();
                for (auto &t : n_outputs) {
                    if (color.color(t) == io.color) continue;
                    std::vector<Node> t_links;
                    for (auto &k : t.inputs()) {
                        auto it = map_origin2sub.find(k);
                        if (it == map_origin2sub.end()) {
                            t_links.emplace_back(k);
                        } else {
                            t_links.emplace_back(it->second);
                        }
                    }
                    Node::Link(t, t_links);
                }
            }
            SubGraph sub;
            sub.m_graph = g;
            sub.m_inputs = sub_inputs;
            sub.m_outputs = io.outputs;
            main.m_sub_graphs.emplace_back(sub);
            main.m_sub_nodes.emplace_back(sub_node[0]);
        }

        auto main_inputs = bottoms;
        std::vector<Node> main_outputs;
        for (auto &n : ends) {
            main_outputs.emplace_back(n.inputs()[0]);
            Node::Link(n, {});
        }
        main.m_module = std::make_shared<Module>();
        main.m_module->load(g, main_outputs);
        main.m_module->sort_inputs(main_inputs);

        return main;
    }

    Splitter::Splitter(SplitRule *rule)
        : m_rule(rule) {

    }
}
