//
// Created by kier on 2018/10/17.
//

#include <runtime/instruction/stack_instruction.h>

#include "runtime/workbench.h"

namespace ts {
    namespace instruction {
        Instruction::shared Stack::push(int i) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                workbench.stack().push(i);
            }, "push(" + std::to_string(i) + ")");
        }

        Instruction::shared Stack::clone(int i) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                workbench.stack().clone(i);
            }, "clone(" + std::to_string(i) + ")");
        }

        Instruction::shared Stack::erase(int i) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                workbench.stack().erase(i);
            }, "erase(" + std::to_string(i) + ")");
        }

        Instruction::shared Stack::ring_shift_left() {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                auto &stack = workbench.stack();
                stack.push(0);
                stack.erase(0);
            }, "<<<(" + std::to_string(1) + ")");
        }

        Instruction::shared Stack::swap(int i, int j) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                auto &stack = workbench.stack();
                Tensor ti = *stack.index(i);
                Tensor tj = *stack.index(j);
                *stack.index(i) = tj;
                *stack.index(j) = ti;
            }, "swap(" + std::to_string(i) + ", " + std::to_string(j) + ")");
        }

        Instruction::shared Stack::erase(int beg, int end) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                workbench.stack().erase(beg, end);
            }, "erase(" + std::to_string(beg) + ", " + std::to_string(end) + ")");
        }

        Instruction::shared Stack::pack(size_t size) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                auto &stack = workbench.stack();
                if (stack.size() < size) {
                    throw Exception(std::string("Can not pack ") + std::to_string(size) + "tensor(s) on stack(size=" + std::to_string(stack.size()) + ")");
                }
                std::vector<Tensor> fields;
                fields.reserve(size);
                int anchor = -int(size);
                while (anchor < 0) {
                    fields.emplace_back(stack.index(anchor));
                    ++anchor;
                }
                Tensor packed_tensor;
                packed_tensor.pack(fields);
                stack.pop(size);
                stack.push(packed_tensor);
            }, "pack(" + std::to_string(size) + ")");
        }

        Instruction::shared Stack::field(size_t index) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                auto &stack = workbench.stack();
                Tensor field = stack.top()->field(index);
                stack.pop();
                stack.push(field);
            }, "field(" + std::to_string(index) + ")");
        }
    }
}

