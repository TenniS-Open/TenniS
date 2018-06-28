//
// Created by seeta on 2018/6/28.
//

#include <cassert>
#include "utils/need.h"
#include "stack/instruction.h"
#include "stack/workbench.h"

namespace ts {

    FunctionInstruction::FunctionInstruction(const Function::shared &func, int nargs, int nresults)
        : m_func(func), m_nargs(nargs), m_nresults(nresults){
        assert(m_func != nullptr);
    }

    void FunctionInstruction::run(Workbench &workbench) {
        auto &stack = workbench.stack();

        assert(stack.size() >= static_cast<size_t>(m_nargs));

        // save base
        stack.push_base(-m_nargs);
        ts::need pop_base(&Stack::pop_base, &stack);

        // call function
        auto return_size = m_func->run(stack);

        assert(return_size == m_nresults);  // must return given sizes
        assert(stack.size() >= static_cast<size_t>(return_size));    // must have enough returns

        // add base
        stack.erase(0, -m_nresults);
    }
}