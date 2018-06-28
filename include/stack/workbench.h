//
// Created by seeta on 2018/5/25.
//

#ifndef TENSORSTACK_STACK_WORKBENCH_H
#define TENSORSTACK_STACK_WORKBENCH_H

#include <memory>
#include <queue>
#include "function.h"
#include "stack.h"
#include "tensor/tensor.h"
#include "instruction.h"

namespace ts {
    class Workbench {
    public:
        using self = Workbench;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        Stack &stack();

        const Stack &stack() const;

        void jump_relative(int shift);

        void jump_absolute(size_t pointer);

        // push args
        void push(const Tensor &arg);

        // clear all stack
        void clear();


    private:
        size_t m_pointer = 0;   // pointer to running function
        std::vector<Instruction::shared> m_program; // running function, program area
        Stack m_stack;  // save running memory, data area
        // Stack m_heap;   // save static area
    };
}


#endif //TENSORSTACK_STACK_WORKBENCH_H
