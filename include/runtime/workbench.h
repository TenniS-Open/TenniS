//
// Created by seeta on 2018/5/25.
//

#ifndef TENSORSTACK_RUNTIME_WORKBENCH_H
#define TENSORSTACK_RUNTIME_WORKBENCH_H

#include <memory>
#include <queue>
#include "operator.h"
#include "stack.h"
#include "core/tensor.h"
#include "instruction.h"
#include "module/module.h"
#include <unordered_map>

namespace ts {
    class Workbench {
    public:
        using self = Workbench;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        template<typename K, typename V>
        using map = std::unordered_map<K, V>;

        explicit Workbench(const ComputingDevice &device);

        Stack &stack() { return *this->m_stack; }

        const Stack &stack() const { return *this->m_stack; }

        void jump_relative(int shift);

        void jump_absolute(size_t pointer);

        // clear all stack
        void clear() { this->m_stack->clear(); }

        // set input
        void input(const std::string &name, const Tensor &tensor);

        void input(int slot, const Tensor &tensor);

        const Tensor &input(const std::string &name) const;

        const Tensor &input(int slot) const;

        // run graph
        void run();

        // get output
        const Tensor &output(const std::string &name) const;

        const Tensor &output(int slot) const;

        // clone an Workbench which can run
        Workbench::shared clone() const;

        static shared Load(const Module::shared &module, const ComputingDevice &device);

    private:
        ComputingDevice m_device;
        size_t m_pointer = 0;   // pointer to running function
        std::vector<Instruction::shared> m_program; // running function, program area
        MemoryController::shared m_static_memory;
        MemoryController::shared m_flow_memory;
        MemoryController::shared m_dynamic_memory;
        Stack::shared m_stack;  // save running memory, data area
        // Stack m_heap;   // save static area
        // map slot, means <tensor'name, tensor's index in stack>
        map<std::string, int> m_map_input_slots;
        map<std::string, int> m_map_output_slots;
        // map tensor, means <tensor's index in stack, tensor>
        std::vector<Tensor> m_inputs;
        std::vector<Tensor> m_outputs;
    };
}


#endif //TENSORSTACK_RUNTIME_WORKBENCH_H
