//
// Created by kier on 2018/11/3.
//

#ifndef TENSORSTACK_RUNTIME_INSTRUCTION_TENSOR_INSTRUCTION_H
#define TENSORSTACK_RUNTIME_INSTRUCTION_TENSOR_INSTRUCTION_H

#include "../instruction.h"

namespace ts {
    namespace instruction {
        // TODO: use InstructionCreator::Register register field operator
        class Tensor {
        public:
            // [-size, +1, e]
            static Instruction::shared pack(size_t size);

            // [-1, +1, e]
            static Instruction::shared field(size_t index);
        };
    }
}


#endif //TENSORSTACK_RUNTIME_INSTRUCTION_TENSOR_INSTRUCTION_H
