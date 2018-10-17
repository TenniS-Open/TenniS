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

        for (auto node : simulator) {
            std::cout << node << std::endl;
        }

        // inverse

        return block;
    }
}