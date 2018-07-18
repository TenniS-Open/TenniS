//
// Created by seeta on 2018/5/25.
//

#include "stack/workbench.h"

namespace ts {
    Workbench::Workbench(const Device &device)
            : m_device(device) {
        this->m_static_memory = std::make_shared<BaseMemoryController>(device);
        this->m_flow_memory = std::make_shared<BaseMemoryController>(device);   // TODO: Make real flow memory controller
        this->m_dynamic_memory = std::make_shared<BaseMemoryController>(device);
        this->m_stack = std::make_shared<Stack>(device, this->m_flow_memory);
        this->m_input = std::make_shared<Stack>(device, this->m_dynamic_memory);
        this->m_output = std::make_shared<Stack>(device, this->m_dynamic_memory);
    }

    void Workbench::run() {
        // clear pre output
        this->m_stack->rebase(0);
        this->m_stack->clear();
        this->m_output->clear();

        // set input
        for (int i = 0; static_cast<size_t >(i) < this->m_input->size(); ++i) {
            auto &arg = *this->m_input->index(i);
            this->m_stack->push(arg);
        }

        // run
        while (m_pointer < m_program.size()) {
            auto &inst = m_program[m_pointer];
            inst->run(*this);
        }

        // set output
        for (int i = 0; static_cast<size_t >(i) < this->m_stack->size(); ++i) {
            auto &arg = *this->m_stack->index(i);
            this->m_output->push(arg);
        }
    }

    Tensor &Workbench::output(int i) {
        return *this->m_stack->index(i);
    }

    const Tensor &Workbench::output(int i) const {
        return *this->m_stack->index(i);
    }

    Tensor &Workbench::input(int i) {
        return *this->m_input->index(i);
    }

    const Tensor &Workbench::input(int i) const {
        return *this->m_input->index(i);
    }
}
