//
// Created by seeta on 2018/6/28.
//

#ifndef TENSORSTACK_STACK_INSTRUCTION_H
#define TENSORSTACK_STACK_INSTRUCTION_H

#include <memory>
#include "operator.h"

namespace ts {

    class Workbench;

    class Instruction {
    public:
        using self = Instruction;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        virtual void run(Workbench &workbench) = 0;
    };

    class OperatorInstruction : public Instruction {
    public:
        using self = OperatorInstruction;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using supper = Instruction;

        explicit OperatorInstruction(const Operator::shared &func, int nargs, int nresults);

        void run(Workbench &workbench) override ;

    private:
        Operator::shared m_func = nullptr;
        int m_nargs = 0;
        int m_nresults = 0;
    };

}


#endif //TENSORSTACK_STACK_INSTRUCTION_H
