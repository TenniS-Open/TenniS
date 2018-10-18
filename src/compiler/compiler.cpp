//
// Created by kier on 2018/10/17.
//

#include "compiler/compiler.h"
#include "module/module.h"
#include "runtime/instruction.h"

#include <deque>
#include <set>
#include <map>
#include "runtime/instruction/istack.h"
#include "global/operator_factory.h"


namespace ts {

    Compiler::Compiler(const ComputingDevice &computing_device)
            : m_computing_device(computing_device) {
    }

    InstructionBlock Compiler::compile(const std::vector<Node> &inputs, const std::vector<Node> &outputs) {
        InstructionBlock block;
        block.nargs = int(inputs.size());
        block.nresults = int(outputs.size());

        // transform graph, make other graph from different framework to TS framework

        // compile graph, check node to computing device support nodes
        // 考虑处理inplace操作符，加入copy节点，保证inplace操作不会对其他节点造成影响

        // convert graph to instructions
        std::deque<Node> simulator;
        std::map<Node, size_t> working_nodes;
        size_t unsolved_node_count = 0;

        /**
         * \brief add node to simulator
         * \param node node ready to push
         */
        auto simulator_push = [&](Node node) {
            auto &op = node.ref<OP>();
            size_t i = simulator.size();
            auto it = working_nodes.find(node);
            if (it == working_nodes.end()) {
                working_nodes.insert(std::make_pair(node, i));
            } else {
                if (i < it->second) it->second = i;
            }
            if (op.op != OP::IN) {
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
            auto &op = node.ref<OP>();
            size_t i = simulator.size() - 1;
            auto it = working_nodes.find(node);
            if (it == working_nodes.end() && it->second == i) {
                working_nodes.erase(node);
            }
            if (op.op != OP::IN) {
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
                nodei_it->second = index_i;
            }
            auto nodej_it = working_nodes.find(nodej);
            if (nodej_it != working_nodes.end() && nodej_it->second == index_j) {
                nodej_it->second = index_j;
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
                auto &op = node.ref<OP>();
                if (op.op != OP::IN) return i;
            }
            return -1;
        };

        for (auto &node : outputs) simulator_push(node);

        // TODO: checking inplace operator converting
        while (unsolved_node_count) {
            auto node = simulator.back();
            auto op = node.ref<OP>();
            // check if node are same node
            auto i = simulator.size() - 1;
            auto it = working_nodes.find(node);
            if (it != working_nodes.end()) {
                if (it->second < i) {
                    block.instructions.push_back(instruction::Stack::push(int(it->second)));
                    simulator_pop();
                    continue;
                }
            }

            if (op.op == OP::IN) {
                auto j = simulator_find_last_unsolved_node_index();
                assert(j >= 0);
                block.instructions.push_back(instruction::Stack::swap(int(i), int(j)));
                simulator_swap(int(i), int(j));
                continue;
            }

            // query operator
            auto creator = QueryOperatorCreator(m_computing_device.type(), op.op);
            if (creator == nullptr) throw Exception("Not supported operator " + op.op);
            block.instructions.push_back(
                    std::make_shared<OperatorInstruction>(creator(), node.inputs().size(), node.outputs().size()));
            simulator_pop();
            for (auto &input : node.inputs()) simulator_push(input);
        }
        // check inputs
        std::set<Node> have_inputs(inputs.begin(), inputs.end());
        for (auto &node : simulator) {
            if (have_inputs.find(node) == have_inputs.end()) {
                throw Exception("Can not access input node: " + node.str());
            }
        }
        // build inputs
        // -.1 check if inputs satisfied
        bool satisfied = true;
        if (simulator.size() == inputs.size()) {
            for (size_t i = 0; i < simulator.size(); ++i) {
                if (simulator[i] != inputs[i]) {
                    satisfied = false;
                    break;
                }
            }
        }
        if (!satisfied) {
            std::map<Node, size_t> working_input_nodes;
            for (auto &node : inputs) working_input_nodes.insert(std::make_pair(node, working_input_nodes.size()));
            block.instructions.push_back(instruction::Stack::erase(0, -int(simulator.size())));
            for (auto it = simulator.rbegin(); it != simulator.rend(); ++it) {
                block.instructions.push_back(instruction::Stack::push(int(working_input_nodes[*it])));
            }
            simulator.clear();
            simulator.insert(simulator.begin(), inputs.begin(), inputs.end());
        }

        for (auto node : simulator) {
            std::cout << node << std::endl;
        }

        // reduce
        // 删除冗余的push，是否有必要，push的成本很低
        // inplace operator 是不是可以检测operator，如果是inplace操作，就把push换成clone。或者不支持inplace操作，最简单了。
        // 思考一下怎么处理额，可以在图的编译阶段，如果支持inplace操作，就插入一个copy节点。

        // inverse

        return block;
    }
}