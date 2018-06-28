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

        explicit Workbench(const Device &device)
                : m_stack(device), m_input(device) {}

        explicit Workbench(const Device &device, const MemoryController::shared &controller)
                : m_stack(device, controller), m_input(device, controller) {}

        Stack &stack() {return this->m_stack;}

        const Stack &stack() const{return this->m_stack;}

        void jump_relative(int shift) {this->m_pointer += shift;}

        void jump_absolute(size_t pointer) {this->m_pointer = pointer;}

        // clear all stack
        void clear() {this->m_stack.clear();}

        // set input
        Tensor &input(int i);
        const Tensor &input(int i) const;

        void run();

        // set output
        Tensor &output(int i);
        const Tensor &output(int i) const;

    private:
        size_t m_pointer = 0;   // pointer to running function
        std::vector<Instruction::shared> m_program; // running function, program area
        Stack m_stack;  // save running memory, data area
        // Stack m_heap;   // save static area
        Stack m_input;  // saving input
    };
}


#endif //TENSORSTACK_STACK_WORKBENCH_H
