//
// Created by seeta on 2018/6/28.
//

#include <cassert>
#include <runtime/instruction.h>

#include "utils/need.h"
#include "runtime/instruction.h"
#include "runtime/workbench.h"

namespace ts {

    OperatorInstruction::OperatorInstruction(const Operator::shared &func, int nargs, int nresults)
            : m_func(func), m_nargs(nargs), m_nresults(nresults) {
        assert(m_func != nullptr);
    }

    void OperatorInstruction::run(Workbench &workbench) {
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

    void StackInstruction::run(Workbench &workbench) {
        this->run(workbench.stack());
    }

    LambdaInstruction::LambdaInstruction(const LambdaInstruction::Lambda &lambda)
            : m_lambda(lambda) {
    }

    void LambdaInstruction::run(Workbench &workbench) {
        m_lambda(workbench);
    }
}