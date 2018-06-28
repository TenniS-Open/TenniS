//
// Created by seeta on 2018/6/28.
//

#ifndef TENSORSTACK_STACK_INSTRUCTION_H
#define TENSORSTACK_STACK_INSTRUCTION_H

#include <memory>
#include "function.h"

namespace ts {

    class Workbench;

    class Instruction {
    public:
        using self = Instruction;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        virtual void run(Workbench &workbench) = 0;
    };

    class FunctionInstruction : public Instruction {
    public:
        using self = FunctionInstruction;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using supper = Instruction;

        explicit FunctionInstruction(const Function::shared &func, int nargs, int nresults);

        void run(Workbench &workbench) override ;

    private:
        Function::shared m_func = nullptr;
        int m_nargs = 0;
        int m_nresults = 0;
    };

}


#endif //TENSORSTACK_STACK_INSTRUCTION_H
