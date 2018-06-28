//
// Created by seeta on 2018/5/25.
//

#include "stack/workbench.h"

namespace ts {
    void Workbench::run() {
        // clear pre output
        this->m_stack.rebase(0);
        this->m_stack.clear();

        // set input
        for (int i = 0; static_cast<size_t >(i) < this->m_input.size(); ++i) {
            auto &arg = *this->m_input.index(i);
            this->m_stack.push(arg);
        }

        // run
        while (m_pointer < m_program.size()) {
            auto &inst = m_program[m_pointer];
            inst->run(*this);
        }
    }

    Tensor &Workbench::output(int i) {
        return *this->m_stack.index(i);
    }

    const Tensor &Workbench::output(int i) const {
        return *this->m_stack.index(i);
    }

    Tensor &Workbench::input(int i) {
        return *this->m_input.index(i);
    }

    const Tensor &Workbench::input(int i) const {
        return *this->m_input.index(i);
    }
}
