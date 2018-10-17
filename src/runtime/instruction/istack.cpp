//
// Created by kier on 2018/10/17.
//

#include <runtime/instruction/istack.h>

#include "runtime/workbench.h"

namespace ts {
    namespace instruction {
        Instruction::shared Stack::push(int i) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                workbench.stack().push(i);
            });
        }

        Instruction::shared Stack::clone(int i) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                workbench.stack().clone(i);
            });
        }

        Instruction::shared Stack::erase(int i) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                workbench.stack().erase(i);
            });
        }

        Instruction::shared Stack::ring_shift_left() {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                auto stack = workbench.stack();
                stack.push(0);
                stack.erase(0);
            });
        }

        Instruction::shared Stack::swap(int i, int j) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                auto stack = workbench.stack();
                Tensor ti = *stack.index(i);
                Tensor tj = *stack.index(j);
                *stack.index(i) = tj;
                *stack.index(j) = ti;
            });
        }
    }
}

