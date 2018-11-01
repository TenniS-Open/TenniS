//
// Created by kier on 2018/10/17.
//

#ifndef TENSORSTACK_RUNTIME_INSTRUCTION_ISTACK_H
#define TENSORSTACK_RUNTIME_INSTRUCTION_ISTACK_H

#include "../instruction.h"

namespace ts {
    namespace instruction {
        class Stack {
        public:
            static Instruction::shared push(int i);
            static Instruction::shared clone(int i);
            static Instruction::shared erase(int i);
            static Instruction::shared erase(int beg, int end);
            static Instruction::shared ring_shift_left();
            static Instruction::shared swap(int i, int j);
        };
    }
}


#endif //TENSORSTACK_RUNTIME_INSTRUCTION_ISTACK_H
