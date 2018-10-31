//
// Created by kier on 2018/10/17.
//

#ifndef TENSORSTACK_COMPILER_COMPILER_H
#define TENSORSTACK_COMPILER_COMPILER_H

#include "runtime/instruction.h"
#include "module/graph.h"

namespace ts {

    class InstructionBlock
    {
    public:
        int nargs = 0;
        int nresults = 0;
        std::vector<Instruction::shared> instructions;
    };

    class Compiler {
    public:
        using self = Compiler;

        explicit Compiler(const ComputingDevice &computing_device);

        InstructionBlock compile(const std::vector<Node> &inputs, const std::vector<Node> &outputs);
        Instruction::shared convert_operator_instruction(const Node &node);
    private:
        ComputingDevice m_computing_device;
    };

}


#endif //TENSORSTACK_COMPILER_COMPILER_H
