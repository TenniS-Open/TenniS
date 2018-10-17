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
        for (auto &node : outputs) simulator.push_back(node);
        std::map<Node, size_t> working_inptus;

        auto add_node_to_simulator = [&](Node node) {
            auto op = node.ref<OP>();
            size_t i = simulator.size();
            if (op.op == OP::IN) {
                auto it = working_inptus.find(node);
                if (it == working_inptus.end()) {
                    working_inptus.insert(std::make_pair(node, i));
                } else {
                    if (i < it->second) it->second = i;
                }
            }
            simulator.push_back(node);
        };

        auto pop_simulator = [&]() {
            simulator.pop_back();
        };

        auto ring_shift_right_simulator = [&]() {
            simulator.push_front(simulator.back());
            simulator.pop_back();
        };

        while (!simulator.empty()) {
            if (simulator.size() == working_inptus.size()) {    // check to all inputs
                // check if all reminds node are input nodes
                std::set<Node> simulator_nodes;
                for (auto &node : simulator) simulator_nodes.insert(node);
                if (simulator_nodes.size() == working_inptus.size()) break;
            }
            auto node = simulator.back();
            auto op = node.ref<OP>();
            if (op.op == OP::IN) {
                auto it = working_inptus.find(node);
                if (it != working_inptus.end()) {   // input already working in simulator
                    // add copy inst
                    block.instructions.push_back(instruction::Stack::push(int(it->second)));
                    pop_simulator();
                    break;
                } else {
                    // add left roll stack
                    block.instructions.push_back(instruction::Stack::ring_shift_left());
                    ring_shift_right_simulator();
                    break;
                }
            }
            // query operator
            auto creator = QueryOperatorCreator(m_computing_device.type(), op.op);
            if (creator == nullptr) throw Exception("Not supported operator " + op.op);
            block.instructions.push_back(std::make_shared<OperatorInstruction>(creator(), node.inputs().size(), node.outputs().size()));
            pop_simulator();
            for (auto &input : node.inputs()) add_node_to_simulator(input);
        }
        // check inputs

        return block;
    }
}