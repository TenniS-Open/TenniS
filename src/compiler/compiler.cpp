//
// Created by kier on 2018/10/17.
//

#include "compiler/compiler.h"
#include "module/module.h"
#include "runtime/instruction.h"

#include <deque>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <cassert>
#include <algorithm>

#include "runtime/instruction/istack.h"
#include "global/operator_factory.h"


namespace ts {
    template <typename K, typename V>
    using map = std::unordered_map<K, V>;
    template <typename K>
    using set = std::unordered_set<K>;


    Compiler::Compiler(const ComputingDevice &computing_device)
            : m_computing_device(computing_device) {
    }

    // TODO: speed this function up
    static map<Node, set<Node>> build_node_refs(const std::vector<Node> &nodes) {
        map<Node, int> map_node_depth;
        std::deque<Node> node_walker; // top_down

        for (auto &node : nodes) {
            auto it = map_node_depth.find(node);
            if (it != map_node_depth.end()) continue;
            node_walker.push_back(node);
            map_node_depth.insert(std::make_pair(node, 1));
        }

        while (!node_walker.empty()) {
            auto node = node_walker.front();
            node_walker.pop_front();
            auto depth = map_node_depth[node];
            for (auto &input : node.inputs()) {
                auto input_depth_pair = map_node_depth.find(input);
                if (input_depth_pair == map_node_depth.end()) {
                    map_node_depth.insert(std::make_pair(input, depth + 1));
                    node_walker.push_back(input);
                } else {
                    auto this_input_depth = depth + 1;
                    if (input_depth_pair->second < this_input_depth) {
                        input_depth_pair->second = this_input_depth;
                    }
                }
            }
        }

        std::vector<std::pair<Node, int>> computation_schedule(map_node_depth.begin(), map_node_depth.end());
        std::sort(computation_schedule.begin(), computation_schedule.end(),
                [](const std::pair<Node, int> &lhs, const std::pair<Node, int> &rhs){
            return lhs.second > rhs.second;
        });

        // build refs
        map<Node, set<Node>> map_node_refs;
        for (auto &node : computation_schedule) {
            set<Node> refs;
            for (auto &input : node.first.inputs()) {
                refs.insert(input);
                auto &input_refs = map_node_refs[input];
                refs.insert(input_refs.begin(), input_refs.end());
            }
            map_node_refs.insert(std::make_pair(node.first, std::move(refs)));
        }

        return map_node_refs;
    }

    Instruction::shared Compiler::convert_operator_instruction(const Node &node) {
        auto &bubble = node.ref<Bubble>();
        auto creator = QueryOperatorCreator(m_computing_device.type(), bubble.op());
        if (creator == nullptr) throw Exception("Not supported operator " + bubble.op());
        std::string description = bubble.op() + "(in=" + std::to_string(node.inputs().size()) + ", out=" +
                                  std::to_string(1) + ")";
        auto op = creator();
        for (auto &param : bubble.params()) {
            op->set(param.first, param.second);
        }
        // TODO: check param if valid
        return std::make_shared<OperatorInstruction>(op, node.inputs().size(), 1, description);
    }

    // TODO: inputs only support Parameter, try support other op
    InstructionBlock Compiler::compile(const std::vector<Node> &inputs, const std::vector<Node> &outputs) {
        InstructionBlock block;
        block.nargs = int(inputs.size());
        block.nresults = int(outputs.size());

        // transform graph, make other graph from different framework to TS framework

        // compile graph, check node to computing device support nodes
        // 考虑处理inplace操作符，加入copy节点，保证inplace操作不会对其他节点造成影响

        // build refs
        // save if compile a node, whose nodes needed directly or indirectly
        map<Node, set<Node>> map_node_refs = build_node_refs(outputs);

        // convert graph to instructions
        std::deque<Node> simulator;
        map<Node, size_t> working_nodes;
        size_t unsolved_node_count = 0;

        /**
         * \brief add node to simulator
         * \param node node ready to push
         */
        auto simulator_push = [&](Node node) {
            auto &bubble = node.ref<Bubble>();
            size_t i = simulator.size();
            auto it = working_nodes.find(node);
            if (it == working_nodes.end()) {
                working_nodes.insert(std::make_pair(node, i));
            } else {
                if (i < it->second) it->second = i;
            }
            if (bubble.op() != Bubble::Parameter) {
                ++unsolved_node_count;
            }
            simulator.push_back(node);
        };

        /**
         * \brief pop node from simulator
         */
        auto simulator_pop = [&]() {
            if (simulator.empty()) return;
            auto node = simulator.back();
            auto &bubble = node.ref<Bubble>();
            size_t i = simulator.size() - 1;
            auto it = working_nodes.find(node);
            if (it != working_nodes.end() && it->second == i) {
                working_nodes.erase(node);
            }
            if (bubble.op() != Bubble::Parameter) {
                --unsolved_node_count;
            }
            simulator.pop_back();
        };

        /**
         * \brief swap nodes at i and j
         * \param i
         * \param j
         * \param only used in swap last in node and last not in node
         */
        auto simulator_swap = [&](int i, int j) {
            // pop front
            auto index_i = i >= 0 ? size_t(i) : size_t(int64_t(simulator.size()) + i);
            auto index_j = j >= 0 ? size_t(j) : size_t(int64_t(simulator.size()) + j);

            auto nodei = simulator[index_i];
            auto nodej = simulator[index_j];

            auto nodei_it = working_nodes.find(nodei);
            if (nodei_it != working_nodes.end() && nodei_it->second == index_i) {
                nodei_it->second = index_j;
            }
            auto nodej_it = working_nodes.find(nodej);
            if (nodej_it != working_nodes.end() && nodej_it->second == index_j) {
                nodej_it->second = index_i;
            }

            simulator[index_i] = nodej;
            simulator[index_j] = nodei;
        };

        /**
         * \brief find last unsolved node index
         * \return found index
         * \note return -1 if failed
         */
        auto simulator_find_last_unsolved_node_index = [&]() -> int64_t {
            int64_t i = int64_t(simulator.size()) - 1;
            while (i >= 0) {
                auto &node = simulator[i];
                auto &bubble = node.ref<Bubble>();
                if (bubble.op() != Bubble::Parameter) return i;
                --i;
            }
            return -1;
        };

        /**
         * \brief find last node ref the given node `ref`
         * \param ref give node
         * \return found index
         * \note return -1 if failed
         */
        auto simulator_find_last_ref_node_index = [&](Node ref) -> int64_t {
            int64_t i = int64_t(simulator.size()) - 1;
            while (i >= 0) {
                auto &node = simulator[i];
                auto &refs = map_node_refs[node];
                if (refs.find(ref) != refs.end()) return i;
                --i;
            }
            return -1;
        };

        for (auto &node : outputs) simulator_push(node);

        // TODO: checking inplace operator converting
        // TODO: check if there are some repeating computing brach
        while (unsolved_node_count) {
            auto node = simulator.back();
            auto bubble = node.ref<Bubble>();
            // case1: check if node are same node
            auto i = simulator.size() - 1;
            auto it = working_nodes.find(node);
            if (it != working_nodes.end()) {
                if (it->second < i) {
                    block.instructions.push_back(instruction::Stack::push(int(it->second)));
                    simulator_pop();
                    continue;
                }
            }

            // case2-0: use Const
            if (bubble.op() == Bubble::Const) {
                // TODO: add const getter
            }

            // case2-1: use Variable
            if (bubble.op() == Bubble::Variable) {
                throw Exception(std::string("Not support ") + Bubble::Variable + " in this version.");
            }

            // case2: save input nodes, move last unsolved node to top
            if (bubble.op() == Bubble::Parameter) {
                auto j = simulator_find_last_unsolved_node_index();
                assert(j >= 0);
                block.instructions.push_back(instruction::Stack::swap(int(i), int(j)));
                simulator_swap(int(i), int(j));
                continue;
            }

            // case3: check if this node will be compute later, if true, then swap it's son to top
            auto last_ref_node_index = simulator_find_last_ref_node_index(node);
            if (last_ref_node_index >= 0) {
                auto j = last_ref_node_index;
                block.instructions.push_back(instruction::Stack::swap(int(i), int(j)));
                simulator_swap(int(i), int(j));
                continue;
            }

            // case4: found a node need to be compute. query operator
            block.instructions.push_back(convert_operator_instruction(node));
            simulator_pop();
            for (auto &input : node.inputs()) simulator_push(input);
        }
        // check inputs
        set<Node> have_inputs(inputs.begin(), inputs.end());
        for (auto &node : simulator) {
            if (have_inputs.find(node) == have_inputs.end()) {
                throw Exception("Can not access input node: " + node.str());
            }
        }

        // build inputs
        // -.1 check if inputs satisfied
        bool satisfied = false;
        if (simulator.size() == inputs.size()) {
            satisfied = true;
            for (size_t i = 0; i < simulator.size(); ++i) {
                if (simulator[i] != inputs[i]) {
                    satisfied = false;
                    break;
                }
            }
        }
        if (!satisfied) {
            map<Node, size_t> working_input_nodes;
            for (auto &node : inputs) working_input_nodes.insert(std::make_pair(node, working_input_nodes.size()));
            block.instructions.push_back(instruction::Stack::erase(0, -int(simulator.size())));
            for (auto it = simulator.rbegin(); it != simulator.rend(); ++it) {
                block.instructions.push_back(instruction::Stack::push(int(working_input_nodes[*it])));
            }
            simulator.clear();
            simulator.insert(simulator.begin(), inputs.begin(), inputs.end());
        }

        // reduce
        // 删除冗余的push，是否有必要，push的成本很低
        // inplace operator 是不是可以检测operator，如果是inplace操作，就把push换成clone。或者不支持inplace操作，最简单了。
        // 思考一下怎么处理额，可以在图的编译阶段，如果支持inplace操作，就插入一个copy节点。

        // reverse
        std::reverse(block.instructions.begin(), block.instructions.end());

        return block;
    }
}