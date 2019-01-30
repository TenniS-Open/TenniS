//
// Created by kier on 2018/6/28.
//


#include <runtime/instruction.h>

#include "utils/need.h"
#include "runtime/instruction.h"
#include "runtime/workbench.h"

#include "board/profiler.h"
#include "core/tensor_builder.h"
#include "module/bubble.h"

namespace ts {

    OperatorInstruction::OperatorInstruction(const Operator::shared &func, int nargs, int nresults)
            : m_func(func), m_nargs(nargs), m_nresults(nresults) {
        TS_AUTO_CHECK(m_func != nullptr);
    }

    Later profiler_run(const Operator::shared &op) {
        if (!profiler_on()) return Later();
        std::ostringstream oss;
        oss << "op(%04d):"
            << tensor::to_string(op->get(Bubble::RetentionParam::op)) << ":"
            << tensor::to_string(op->get(Bubble::RetentionParam::name));
        return profiler_serial_timer(oss.str());
    }

    void OperatorInstruction::run(Workbench &workbench) {
        auto &stack = workbench.stack();

        TS_AUTO_CHECK(stack.size() >= static_cast<size_t>(m_nargs));

        // save base
        stack.push_base(-m_nargs);
        ts::need pop_base(&Stack::pop_base, &stack);

        // call function
        int return_size = 0;
        {
            auto _timer = profiler_run(this->m_func);
            return_size = m_func->run(stack);
        }

        (void)(return_size);

        TS_AUTO_CHECK(return_size == m_nresults);  // must return given sizes
        TS_AUTO_CHECK(stack.size() >= static_cast<size_t>(return_size));    // must have enough returns

        // add base
        stack.erase(0, -m_nresults);
    }

    OperatorInstruction::OperatorInstruction(const Operator::shared &func, int nargs, int nresults,
                                             const std::string &description)
            : m_func(func), m_nargs(nargs), m_nresults(nresults), m_description(description) {
        TS_AUTO_CHECK(m_func != nullptr);
    }

    std::string OperatorInstruction::str() const {
        if (m_description.empty()) return supper::str();
        std::ostringstream oss;
        oss << "<Operator: " << m_description << ">";
        return oss.str();
    }

    OperatorInstruction::shared OperatorInstruction::clone() const {
        if (m_creator == nullptr) {
            TS_LOG_ERROR << "Can not clone operator without creator bind" << eject;
        }
        auto op = m_creator();
        for (auto &param : m_func->params()) {
            op->set(param.first, param.second);
        }
        op->init();
        auto dolly = std::make_shared<OperatorInstruction>(op, m_nargs, m_nresults, m_description);
        dolly->m_creator = m_creator;
        return std::move(dolly);
    }

    void OperatorInstruction::bind_creator(OperatorCreator::function creator) {
        m_creator = std::move(creator);
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
        oss << "<Const: @" << m_data_index << ">";
        return oss.str();
    }
}