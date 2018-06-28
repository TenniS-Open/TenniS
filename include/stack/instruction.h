//
// Created by seeta on 2018/6/28.
//

#ifndef TENSORSTACK_STACK_INSTRUCTION_H
#define TENSORSTACK_STACK_INSTRUCTION_H

#include <memory>

namespace ts {

    class Workbench;
    class Instruction {
    public:
        using self = Instruction;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        virtual int run(Workbench &workbench) = 0;
    };

}


#endif //TENSORSTACK_STACK_INSTRUCTION_H
