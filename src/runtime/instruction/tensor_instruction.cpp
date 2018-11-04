//
// Created by kier on 2018/11/3.
//

#include "runtime/instruction/tensor_instruction.h"

#include "runtime/workbench.h"

#include "runtime/instruction/instruction_factory.h"
#include "utils/static.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace instruction {
        Instruction::shared Tensor::pack(size_t size) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench) {
                auto &stack = workbench.stack();
                if (stack.size() < size) {
                    throw Exception(std::string("Can not pack ") + std::to_string(size) + "tensor(s) on stack(size=" +
                                    std::to_string(stack.size()) + ")");
                }
                std::vector<ts::Tensor> fields;
                fields.reserve(size);
                int anchor = -int(size);
                while (anchor < 0) {
                    fields.emplace_back(*stack.index(anchor));
                    ++anchor;
                }
                ts::Tensor packed_tensor;
                packed_tensor.pack(fields);
                stack.pop(size);
                stack.push(packed_tensor);
            }, "pack(" + std::to_string(size) + ")");
        }

        Instruction::shared Tensor::field(size_t index) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench) {
                auto &stack = workbench.stack();
                auto field = stack.top()->field(index);
                stack.pop();
                stack.push(field);
            }, "field(" + std::to_string(index) + ")");
        }

        static std::vector<Instruction::shared> create_instruction_field(const Node &node) {
            assert(node.inputs().size() == 1);

            auto &bubble = node.ref<Bubble>();
            assert(bubble.output_count() == 1);

            std::vector<Instruction::shared> inst;
            int offset = tensor::to_int(bubble.get("offset"));

            assert(offset >= 0);

            inst.push_back(Tensor::field(size_t(offset)));

            return inst;
        };

        static std::vector<Instruction::shared> create_instruction_pack(const Node &node) {
            auto &bubble = node.ref<Bubble>();
            assert(bubble.output_count() == 1);

            std::vector<Instruction::shared> inst;

            inst.push_back(Tensor::pack(node.inputs().size()));

            return inst;
        };
    }
}

TS_STATIC_ACTION(ts::InstructionCreator::Register, "_field", ts::instruction::create_instruction_field);
TS_STATIC_ACTION(ts::InstructionCreator::Register, "_pack", ts::instruction::create_instruction_pack);
