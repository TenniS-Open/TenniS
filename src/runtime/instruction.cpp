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
        (void)(return_size);

        assert(return_size == m_nresults);  // must return given sizes
        assert(stack.size() >= static_cast<size_t>(return_size));    // must have enough returns

        // add base
        stack.erase(0, -m_nresults);
    }

    OperatorInstruction::OperatorInstruction(const Operator::shared &func, int nargs, int nresults,
                                             const std::string &description)
            : m_func(func), m_nargs(nargs), m_nresults(nresults), m_description(description) {
        assert(m_func != nullptr);
    }

    std::string OperatorInstruction::str() const {
        if (m_description.empty()) return supper::str();
        std::ostringstream oss;
        oss << "<Operator: " << m_description << ">";
        return oss.str();
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

    LambdaInstruction::LambdaInstruction(const LambdaInstruction::Lambda &lambda, const std::string &description)
            : m_lambda(lambda), m_description(description) {

    }

    std::string LambdaInstruction::str() const {
        if (m_description.empty()) return supper::str();
        std::ostringstream oss;
        oss << "<Lambda: " << m_description << ">";
        return oss.str();
    }

    std::string Instruction::str() const {
        std::ostringstream oss;
        oss << "<Instruction: " << this << ">";
        return oss.str();
    }

    std::string Instruction::repr() const {
        return this->str();
    }

    DataSagmentInstruction::DataSagmentInstruction(int data_index)
        : m_data_index(data_index) {

    }

    void DataSagmentInstruction::run(Workbench &workbench) {
        workbench.push_data_sagment(m_data_index);
    }

    std::string DataSagmentInstruction::str() const {
        std::ostringstream oss;
        oss << "<Data: @" << m_data_index << ">";
        return oss.str();
    }
}