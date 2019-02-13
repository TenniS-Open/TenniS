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

#include <algorithm>
#include <backend/name.h>

#include "runtime/instruction/instruction_factory.h"
#include "runtime/instruction/stack_instruction.h"
#include "runtime/instruction/tensor_instruction.h"
#include "global/operator_factory.h"
#include "global/memory_device.h"
#include "core/tensor_builder.h"
#include "core/device_context.h"
#include "utils/ctxmgr_lite.h"


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
        auto computation_schedule = Module::list_reference_nodes(nodes);

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

    std::vector<Instruction::shared> Compiler::convert_operator_instruction(const Node &node) {
        auto &bubble = node.bubble();

        // step 1: check inner InstructionCreator
        auto icreator = InstructionCreator::Query(bubble.op());
        if (icreator != nullptr) {
            return icreator(node);
        }
        auto creator = OperatorCreator::Query(m_computing_device.type(), bubble.op(), false);

        if (creator == nullptr) TS_LOG_ERROR << "Not supported operator " << bubble.op() << eject;
        std::string description = bubble.op() + "(in=" + std::to_string(node.inputs().size()) + ", out=" +
                                  std::to_string(1) + ")";
        auto op = creator();

        for (auto &param : bubble.params()) {
            op->set(param.first, param.second);
        }
        try {
            op->init();
        } catch (const Exception &e) {
            TS_LOG_ERROR << "While initializing " << bubble.op() << ":" << bubble.name() << " got Exception: " << e.what() << eject;
        }
        std::vector<Instruction::shared> instructions;
        auto op_inst = std::make_shared<OperatorInstruction>(op, int(node.inputs().size()), int(bubble.output_count()), description);
        op_inst->bind_creator(creator);
        instructions.emplace_back(std::move(op_inst));
        if (bubble.output_count() > 1) {
            instructions.emplace_back(instruction::Tensor::pack(size_t(bubble.output_count())));
        }
        return std::move(instructions);
    }

    // TODO: inputs only support Parameter, try support other op
    InstructionBlock Compiler::compile(const std::vector<Node> &inputs, const std::vector<Node> &outputs) {
        DeviceContext device_context(m_computing_device);
        ctx::bind<DeviceContext> _bind_device_context(device_context);

        InstructionBlock block;
        block.nargs = int(inputs.size());
        block.nresults = int(outputs.size());

        // transform graph, make other graph from different framework to TS framework

        // compile graph, check node to computing device support nodes
        // 考虑处理inplace操作符，加入copy节点，保证inplace操作不会对其他节点造成影响

        // build refs
        // save if compile a node, whose nodes needed directly or indirectly
        map<Node, set<Node>> map_node_refs = build_node_refs(outputs);

        map<Node, int> map_node_data_sagment_index;

        // convert graph to instructions
        std::deque<Node> simulator;
        map<Node, size_t> working_nodes;
        size_t unsolved_node_count = 0;

        /**
         * \brief add node data sagment
         * \return data index
         */
        auto push_data_sagment = [&](Node node) -> int {
            auto node_it = map_node_data_sagment_index.find(node);
            if (node_it != map_node_data_sagment_index.end()) {
                return node_it->second;
            }
            auto &bubble = node.bubble();
            auto value = bubble.get(name::value);
            auto data_index = int(block.data_sagment.size());
            if (bubble.has(name::device)) {
                block.data_sagment.push_back(DeviceTensor(value, tensor::to_string(bubble.get(name::device))));
            } else {
                block.data_sagment.push_back(DeviceTensor(value));
            }
            map_node_data_sagment_index.insert(std::make_pair(node, data_index));
            return data_index;
        };

        /**
         * \brief add node to simulator
         * \param node node ready to push
         */
        auto simulator_push = [&](Node node) {
            auto &bubble = node.bubble();
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
            auto &bubble = node.bubble();
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
                auto &node = simulator[size_t(i)];
                auto &bubble = node.bubble();
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
                auto &node = simulator[size_t(i)];
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
            auto bubble = node.bubble();
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
                int data_index = push_data_sagment(node);
                block.instructions.push_back(std::make_shared<DataSagmentInstruction>(data_index));
                simulator_pop();
                continue;
            }

            // case2-1: use Variable
            if (bubble.op() == Bubble::Variable) {
                TS_LOG_ERROR << "Not support " << Bubble::Variable << " in this version" << eject;
            }

            // case2: save input nodes, move last unsolved node to top
            if (bubble.op() == Bubble::Parameter) {
                auto j = simulator_find_last_unsolved_node_index();
                TS_AUTO_CHECK(j >= 0);
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
            auto operator_instructions = convert_operator_instruction(node);
            for (auto inst_it = operator_instructions.rbegin(); inst_it != operator_instructions.rend(); ++inst_it) {
                block.instructions.push_back(*inst_it);
            }
            simulator_pop();
            for (auto &input : node.inputs()) simulator_push(input);
        }
        // check inputs
        set<Node> have_inputs(inputs.begin(), inputs.end());
        for (auto &node : simulator) {
            if (have_inputs.find(node) == have_inputs.end()) {
                TS_LOG_ERROR << "Can not access input node: " << node.str() << eject;
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