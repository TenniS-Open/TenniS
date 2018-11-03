//
// Created by kier on 2018/11/3.
//

#ifndef TENSORSTACK_RUNTIME_INSTRUCTION_INSTRUCTION_FACTORY_H
#define TENSORSTACK_RUNTIME_INSTRUCTION_INSTRUCTION_FACTORY_H

#include <functional>

#include "../instruction.h"
#include "module/graph.h"

namespace ts {
    // TODO: add instruction factory, query instruction by name
    // Those instructions are cross computing and memory device operator
    using InstructionBuilder = std::function<std::vector<Instruction::shared>(const Node &node)>;

    /**
     * Example of InstructionBuilder
     * @param node node ready to convert to instruction
     * @return an serial of instructions, those can calculate node
     */
    std::vector<Instruction::shared> InstructionBuilderDeclaration(const Node &node);

    /**
     * Query instruction builder of specific op
     * @param op querying op
     * @return InstructionBuilder
     * @note supporting called by threads without calling @sa RegisterInstructionBuilder
     * @note the query should be the Bubble.op
     */
    InstructionBuilder QueryInstructionBuilder(const std::string &op) TS_NOEXCEPT;

    /**
     * Register InstructionBuilder for specific op
     * @param op specific op name @sa Bubble
     * @param builder instruction builder
     * @note only can be called before running @sa QueryInstructionBuilder
     */
    void RegisterInstructionBuilder(const std::string &op, const InstructionBuilder &builder) TS_NOEXCEPT;

    /**
     * No details for this API, so DO NOT call it
     */
    void ClearRegisterInstructionBuilder();
}


#endif //TENSORSTACK_RUNTIME_INSTRUCTION_INSTRUCTION_FACTORY_H
